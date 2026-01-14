# Pooled LSTM that mixes the data from all 12 sensors together; we train one LSTM on them 
# instead of a separate LSTM for each sensor. 

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict

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
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 60
    patience: int = 10
    train_frac: float = 0.70
    val_frac: float = 0.15
    embed_dim: int = 8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
    horizon: int
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
# Dataset (now includes sensor_idx)
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

        # Combine temporal representation + sensor embedding
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, sensor_idx):
        # x: (B, T, F)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # (B, hidden)

        e = self.sensor_emb(sensor_idx)  # (B, embed_dim)

        z = torch.cat([h_last, e], dim=-1)  # (B, hidden+embed)
        y_hat = self.head(z).squeeze(-1)    # (B,)
        return y_hat


# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def eval_metrics(model, loader, device) -> Dict[str, float]:
    model.eval()
    ys, preds = [], []
    for Xb, sb, yb in loader:
        Xb = Xb.to(device)
        sb = sb.to(device)
        yb = yb.to(device)
        pb = model(Xb, sb)
        ys.append(yb.cpu().numpy())
        preds.append(pb.cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(preds) if preds else np.array([])
    if len(y) == 0:
        return {"rmse": np.nan, "mae": np.nan}
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    return {"rmse": rmse, "mae": mae}


def train_one_model(model, train_loader, val_loader, cfg: Config) -> nn.Module:
    device = cfg.device
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for Xb, sb, yb in train_loader:
            Xb = Xb.to(device)
            sb = sb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(Xb, sb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val = eval_metrics(model, val_loader, device)
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
# Main: pooled training
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

    # Map sensor IDs to indices for embedding
    sensors = sorted(df["sn"].dropna().unique().tolist())
    sn_to_idx = {sn: i for i, sn in enumerate(sensors)}

    # Clean each sensor group and store for processing
    per_sensor = {}
    for sn, g in df.groupby("sn"):
        g = g.dropna(subset=feature_cols + [target_col]).copy().reset_index(drop=True)
        if len(g) < (cfg.seq_len + cfg.horizon + 50):
            continue
        per_sensor[sn] = g

    if not per_sensor:
        raise ValueError("No sensors have enough data after cleaning.")

    # ---------
    # Fit ONE global scaler using ONLY TRAIN rows from ALL sensors
    # (prevents leakage and gives consistent scaling across sensors)
    # ---------
    train_rows = []
    for sn, g in per_sensor.items():
        tr_sl, _, _ = split_by_time(len(g), cfg.train_frac, cfg.val_frac)
        train_rows.append(g.loc[tr_sl, feature_cols].to_numpy())
    train_rows = np.vstack(train_rows)

    scaler = StandardScaler().fit(train_rows)

    # ---------
    # Build pooled window datasets
    # ---------
    X_tr_list, s_tr_list, y_tr_list = [], [], []
    X_va_list, s_va_list, y_va_list = [], [], []
    X_te_list, s_te_list, y_te_list = [], [], []

    for sn, g in per_sensor.items():
        tr_sl, va_sl, te_sl = split_by_time(len(g), cfg.train_frac, cfg.val_frac)

        g_scaled = g.copy()
        g_scaled[feature_cols] = scaler.transform(g_scaled[feature_cols].to_numpy())

        X, y = make_windows(g_scaled, feature_cols, target_col, cfg.seq_len, cfg.horizon)
        pred_indices = np.arange(len(y)) + cfg.seq_len  # aligns each window to its prediction row index

        train_mask = (pred_indices >= tr_sl.start) & (pred_indices < tr_sl.stop)
        val_mask   = (pred_indices >= va_sl.start) & (pred_indices < va_sl.stop)
        test_mask  = (pred_indices >= te_sl.start) & (pred_indices < te_sl.stop)

        s_idx = sn_to_idx[sn]
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

    print(f"\n=== POOLED TRAINING | sensors={len(sensors)} | train windows={len(train_ds)} val={len(val_ds)} test={len(test_ds)} ===")
    model = train_one_model(model, train_loader, val_loader, cfg)

    test = eval_metrics(model, test_loader, cfg.device)
    print(f"\nPOOLED TEST RMSE={test['rmse']:.4f} | TEST MAE={test['mae']:.4f}")

    return {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "feature_cols": feature_cols,
        "sn_to_idx": sn_to_idx,
        "metrics_test": test,
        "cfg": cfg,
    }


if __name__ == "__main__":
    df = pd.read_csv("/Users/gavinyu/Desktop/Air Quality Project/meteo_sensor_together_Joe.csv")
    cfg = Config(seq_len=24, horizon=1, hidden_size=64, num_layers=2, embed_dim=8)
    artifacts = train_pooled_lstm_with_embedding(df, cfg)
