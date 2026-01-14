# ============================
# PATCHES for pooled model:
# 4a) Per-sensor evaluation
# 4b) Hyperparam variants (LR, embed_dim, etc.)
# 4c) Huber loss + optional per-sensor target standardization
# ============================

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    seq_len: int = 24
    horizon: int = 1

    batch_size: int = 128
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # 4b: pooled tuning knobs
    embed_dim: int = 16          # try 8, 16, 32
    lr: float = 3e-4             # try 1e-3, 3e-4, 1e-4
    weight_decay: float = 1e-4

    max_epochs: int = 80
    patience: int = 12
    train_frac: float = 0.70
    val_frac: float = 0.15

    # 4c: robustness options
    use_huber: bool = True
    huber_beta: float = 1.0

    # Optional (recommended if sensors have different PM scales)
    standardize_target_per_sensor: bool = False

    seed: int = 42

    # Device
    device: str = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    )

    exclude_sensors: Tuple[str, ...] = ("MOD-00647",)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Utilities
# ----------------------------
def add_wind_sincos(df: pd.DataFrame, wdir_col: str = "wdir") -> pd.DataFrame:
    df = df.copy()
    wdir = pd.to_numeric(df[wdir_col], errors="coerce")
    radians = np.deg2rad(wdir)
    df["wdir_sin"] = np.sin(radians)
    df["wdir_cos"] = np.cos(radians)
    return df


def time_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if "time" in df.columns and df["time"].notna().any():
        df = df.sort_values(["sn", "time"])
    else:
        df["timelocal"] = pd.to_numeric(df["timelocal"], errors="coerce")
        df = df.sort_values(["sn", "timelocal"])
    return df.reset_index(drop=True)


def split_by_time(n: int, train_frac: float, val_frac: float) -> Tuple[slice, slice, slice]:
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def make_windows(
    df_sensor: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = df_sensor[feature_cols].to_numpy(dtype=np.float32)
    target = pd.to_numeric(df_sensor[target_col], errors="coerce").to_numpy(dtype=np.float32)

    valid = ~np.isnan(target)
    feats = feats[valid]
    target = target[valid]

    N = len(target)
    X_list, y_list = [], []
    for i in range(seq_len, N - horizon):
        X_list.append(feats[i - seq_len : i])
        y_list.append(target[i + horizon])

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else np.empty((0,), dtype=np.float32)
    return X, y


# ----------------------------
# Dataset (pooled + sensor id)
# ----------------------------
class PooledSequenceDataset(Dataset):
    """
    Returns:
      X: (seq_len, n_features)
      sensor_idx: int
      y: float
    """
    def __init__(self, X: np.ndarray, sensor_idx: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.sensor_idx = torch.tensor(sensor_idx, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.sensor_idx[idx], self.y[idx]


# ----------------------------
# Model: LSTM + sensor embedding
# ----------------------------
class PooledLSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_sensors: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        embed_dim: int,
    ):
        super().__init__()

        self.sensor_emb = nn.Embedding(n_sensors, embed_dim)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, sensor_idx):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]                         # (B, hidden)
        e = self.sensor_emb(sensor_idx)          # (B, embed_dim)
        z = torch.cat([h_last, e], dim=-1)       # (B, hidden+embed)
        return self.head(z).squeeze(-1)          # (B,)


# ----------------------------
# 4a) Evaluation helpers:
# overall metrics + per-sensor metrics
# ----------------------------
@torch.no_grad()
def eval_overall(model, loader, device, y_unstandardize=None) -> Dict[str, float]:
    model.eval()
    ys, preds = [], []
    for Xb, sb, yb in loader:
        Xb, sb, yb = Xb.to(device), sb.to(device), yb.to(device)
        pb = model(Xb, sb)

        sb_np = sb.cpu().numpy()
        y_np = yb.cpu().numpy()
        p_np = pb.cpu().numpy()

        if y_unstandardize is not None:
            # unstandardize each row using its sensor stats
            y_out = np.empty_like(y_np)
            p_out = np.empty_like(p_np)
            for i, s_i in enumerate(sb_np):
                mu, sd = y_unstandardize[int(s_i)]
                y_out[i] = y_np[i] * sd + mu
                p_out[i] = p_np[i] * sd + mu
            y_np, p_np = y_out, p_out

        ys.append(y_np)
        preds.append(p_np)

    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(preds) if preds else np.array([])
    if len(y) == 0:
        return {"rmse": np.nan, "mae": np.nan}

    return {
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
    }

@torch.no_grad()
def eval_per_sensor(
    model,
    loader,
    device,
    idx_to_sn: Dict[int, str],
    y_unstandardize: Optional[Dict[int, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    If y_unstandardize is provided, it should map sensor_idx -> (mean, std),
    and we will unstandardize predictions/targets before computing metrics.
    """
    model.eval()
    # Collect per-sensor arrays
    by_s = {}  # s -> {"y": [...], "p": [...]}

    for Xb, sb, yb in loader:
        Xb, sb, yb = Xb.to(device), sb.to(device), yb.to(device)
        pb = model(Xb, sb)

        sb_np = sb.cpu().numpy()
        y_np = yb.cpu().numpy()
        p_np = pb.cpu().numpy()

        for s_i, y_i, p_i in zip(sb_np, y_np, p_np):
            if y_unstandardize is not None:
                mu, sd = y_unstandardize[int(s_i)]
                y_i = y_i * sd + mu
                p_i = p_i * sd + mu
            if int(s_i) not in by_s:
                by_s[int(s_i)] = {"y": [], "p": []}
            by_s[int(s_i)]["y"].append(float(y_i))
            by_s[int(s_i)]["p"].append(float(p_i))

    rows = []
    for s_i, d in by_s.items():
        y = np.array(d["y"])
        p = np.array(d["p"])
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        mae = float(np.mean(np.abs(p - y)))
        rows.append({"sn": idx_to_sn[s_i], "rmse": rmse, "mae": mae, "n": len(y)})

    out = pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    out["rmse_rank"] = np.arange(1, len(out) + 1)
    return out


# ----------------------------
# 4c) Target standardization (optional)
# ----------------------------
def compute_sensor_target_stats(per_sensor: Dict[str, pd.DataFrame], target_col: str, cfg: Config) -> Dict[str, Tuple[float, float]]:
    """
    Compute (mean, std) of y on TRAIN portion for each sensor.
    Used if cfg.standardize_target_per_sensor is True.
    """
    stats = {}
    for sn, g in per_sensor.items():
        tr_sl, _, _ = split_by_time(len(g), cfg.train_frac, cfg.val_frac)
        y = pd.to_numeric(g.loc[tr_sl, target_col], errors="coerce").to_numpy(dtype=np.float32)
        y = y[~np.isnan(y)]
        mu = float(np.mean(y))
        sd = float(np.std(y) + 1e-6)
        stats[sn] = (mu, sd)
    return stats


def standardize_y(y: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (y - mu) / sd


# ----------------------------
# Training
# ----------------------------
def train_one_model(model, train_loader, val_loader, cfg: Config) -> nn.Module:
    device = cfg.device
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 4c: Huber loss option
    if cfg.use_huber:
        loss_fn = nn.SmoothL1Loss(beta=cfg.huber_beta)
    else:
        loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for Xb, sb, yb in train_loader:
            Xb, sb, yb = Xb.to(device), sb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(Xb, sb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val = eval_overall(model, val_loader, device)
        print(f"Epoch {epoch:03d} | val RMSE={val['rmse']:.4f} | val MAE={val['mae']:.4f}")

        if val["rmse"] < best_val - 1e-4:
            best_val = val["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------------------
# Main: pooled training (with 4a/4b/4c options)
# ----------------------------
def train_pooled_lstm_with_embedding(df: pd.DataFrame, cfg: Config):
    set_seed(cfg.seed)

    df = time_sort(df)
    df = add_wind_sincos(df)

    target_col = "pm25_mean"
    df["pm25_in"] = pd.to_numeric(df[target_col], errors="coerce")

    feature_cols = [
        "prcp", "pres", "rhum", "temp",
        "wspd", "wdir_sin", "wdir_cos",
        "pm25_in",
    ]

    for c in feature_cols + [target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

   # --- Exclude sensors (e.g., MOD-00647) ---
    excluded = set(cfg.exclude_sensors)

    # Clean per sensor, skipping excluded ones
    per_sensor = {}
    for sn, g in df.groupby("sn"):
        if sn in excluded:
            continue
        g = g.dropna(subset=feature_cols + [target_col]).copy().reset_index(drop=True)
        if len(g) < (cfg.seq_len + cfg.horizon + 50):
            continue
        per_sensor[sn] = g

    if not per_sensor:
        raise ValueError("No sensors have enough data after cleaning (after exclusions).")

    # Sensor index mappings based only on remaining sensors
    sensors = sorted(per_sensor.keys())
    sn_to_idx = {sn: i for i, sn in enumerate(sensors)}
    idx_to_sn = {i: sn for sn, i in sn_to_idx.items()}

    print(f"Excluded sensors: {sorted(list(excluded))}")
    print(f"Using sensors ({len(sensors)}): {sensors}")

    if not per_sensor:
        raise ValueError("No sensors have enough data after cleaning.")

    # 4c (optional): per-sensor target standardization stats (train-only)
    y_stats = None
    if cfg.standardize_target_per_sensor:
        y_stats = compute_sensor_target_stats(per_sensor, target_col, cfg)

    # Fit ONE global scaler on TRAIN rows only (across sensors)
    train_rows = []
    for sn, g in per_sensor.items():
        tr_sl, _, _ = split_by_time(len(g), cfg.train_frac, cfg.val_frac)
        train_rows.append(g.loc[tr_sl, feature_cols].to_numpy())
    train_rows = np.vstack(train_rows)
    scaler = StandardScaler().fit(train_rows)

    # Build pooled windows
    X_tr_list, s_tr_list, y_tr_list = [], [], []
    X_va_list, s_va_list, y_va_list = [], [], []
    X_te_list, s_te_list, y_te_list = [], [], []

    for sn, g in per_sensor.items():
        tr_sl, va_sl, te_sl = split_by_time(len(g), cfg.train_frac, cfg.val_frac)

        g_scaled = g.copy()
        g_scaled[feature_cols] = scaler.transform(g_scaled[feature_cols].to_numpy())

        X, y = make_windows(g_scaled, feature_cols, target_col, cfg.seq_len, cfg.horizon)
        pred_indices = np.arange(len(y)) + cfg.seq_len

        train_mask = (pred_indices >= tr_sl.start) & (pred_indices < tr_sl.stop)
        val_mask   = (pred_indices >= va_sl.start) & (pred_indices < va_sl.stop)
        test_mask  = (pred_indices >= te_sl.start) & (pred_indices < te_sl.stop)

        s_idx = sn_to_idx[sn]

        # 4c (optional): standardize y per sensor using TRAIN stats
        if cfg.standardize_target_per_sensor:
            mu, sd = y_stats[sn]
            y = standardize_y(y, mu, sd)

        s_train = np.full((train_mask.sum(),), s_idx, dtype=np.int64)
        s_val   = np.full((val_mask.sum(),), s_idx, dtype=np.int64)
        s_test  = np.full((test_mask.sum(),), s_idx, dtype=np.int64)

        X_tr_list.append(X[train_mask]); y_tr_list.append(y[train_mask]); s_tr_list.append(s_train)
        X_va_list.append(X[val_mask]);   y_va_list.append(y[val_mask]);   s_va_list.append(s_val)
        X_te_list.append(X[test_mask]);  y_te_list.append(y[test_mask]);  s_te_list.append(s_test)

    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    s_train = np.concatenate(s_tr_list, axis=0)

    X_val = np.concatenate(X_va_list, axis=0)
    y_val = np.concatenate(y_va_list, axis=0)
    s_val = np.concatenate(s_va_list, axis=0)

    X_test = np.concatenate(X_te_list, axis=0)
    y_test = np.concatenate(y_te_list, axis=0)
    s_test = np.concatenate(s_te_list, axis=0)

    train_ds = PooledSequenceDataset(X_train, s_train, y_train)
    val_ds   = PooledSequenceDataset(X_val, s_val, y_val)
    test_ds  = PooledSequenceDataset(X_test, s_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = PooledLSTMRegressor(
        n_features=len(feature_cols),
        n_sensors=len(sensors),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        embed_dim=cfg.embed_dim,
    )

    print(
        f"\n=== POOLED TRAINING | sensors={len(sensors)} "
        f"| train windows={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"| lr={cfg.lr} embed_dim={cfg.embed_dim} huber={cfg.use_huber} y_std={cfg.standardize_target_per_sensor} ==="
    )

    model = train_one_model(model, train_loader, val_loader, cfg)

    # 4a: per-sensor metrics (unstandardize if needed)
    # 4a: unstandardization map (only used when y was standardized per sensor)
    y_unstd = None
    if cfg.standardize_target_per_sensor:
        # Map sensor_idx -> (mu, sd)
        y_unstd = {sn_to_idx[sn]: y_stats[sn] for sn in y_stats.keys()}

    # Overall test metrics (will unstandardize if y_unstd is not None)
    test_overall = eval_overall(model, test_loader, cfg.device, y_unstd)

    # Per-sensor table (also unstandardizes if provided)
    test_by_sensor = eval_per_sensor(model, test_loader, cfg.device, idx_to_sn, y_unstandardize=y_unstd)

    print(f"\nPOOLED TEST OVERALL RMSE={test_overall['rmse']:.4f} | MAE={test_overall['mae']:.4f}\n")
    print("Top 5 worst sensors by RMSE:")
    print(test_by_sensor.head(5).to_string(index=False))
    print("\nTop 5 best sensors by RMSE:")
    print(test_by_sensor.tail(5).to_string(index=False))

    return {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "feature_cols": feature_cols,
        "sn_to_idx": sn_to_idx,
        "idx_to_sn": idx_to_sn,
        "metrics_test_overall": test_overall,
        "metrics_test_by_sensor": test_by_sensor,
        "cfg": cfg,
    }


# ----------------------------
# 4b) Run multiple variants
# ----------------------------
def run_variants(csv_path: str):
    df = pd.read_csv(csv_path)

    variants = [
        # Baseline pooled (your previous-ish)
        Config(embed_dim=8,  lr=1e-3, use_huber=False, standardize_target_per_sensor=False),

        # 4b: lower LR + bigger embedding
        Config(embed_dim=16, lr=3e-4, use_huber=False, standardize_target_per_sensor=False),
        Config(embed_dim=32, lr=3e-4, use_huber=False, standardize_target_per_sensor=False),

        # 4c: add Huber
        Config(embed_dim=16, lr=3e-4, use_huber=True,  standardize_target_per_sensor=False),
        Config(embed_dim=32, lr=3e-4, use_huber=True,  standardize_target_per_sensor=False),

        # 4c: per-sensor target standardization (often helps pooled a lot)
        Config(embed_dim=16, lr=3e-4, use_huber=True,  standardize_target_per_sensor=True),
        Config(embed_dim=32, lr=3e-4, use_huber=True,  standardize_target_per_sensor=True),
    ]

    all_results = []
    for i, cfg in enumerate(variants, 1):
        print(f"\n\n==================== Variant {i}/{len(variants)} ====================")
        artifacts = train_pooled_lstm_with_embedding(df, cfg)
        m = artifacts["metrics_test_overall"]
        all_results.append({
            "variant": i,
            "embed_dim": cfg.embed_dim,
            "lr": cfg.lr,
            "huber": cfg.use_huber,
            "y_std": cfg.standardize_target_per_sensor,
            "test_rmse": m["rmse"],
            "test_mae": m["mae"],
        })

    summary = pd.DataFrame(all_results).sort_values("test_rmse").reset_index(drop=True)
    print("\n\n===== SUMMARY (sorted by test_rmse) =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    # Option A: run ONE configuration
    # df = pd.read_csv("your_data.csv")
    # cfg = Config(embed_dim=16, lr=3e-4, use_huber=True, standardize_target_per_sensor=True)
    # artifacts = train_pooled_lstm_with_embedding(df, cfg)

    # Option B: run a sweep of variants
    run_variants("/Users/gavinyu/Desktop/Air Quality Project/meteo_sensor_together_Joe.csv")
