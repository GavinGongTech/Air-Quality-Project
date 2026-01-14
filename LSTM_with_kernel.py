# Spatial baseline (multi-scale + wind-anisotropic): kernel-weighted neighbor features into pooled LSTM
# Minimal working script: fixes time_col order + uses wind-anisotropic features + prevents LOSO leakage

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

    embed_dim: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4

    max_epochs: int = 80
    patience: int = 12
    train_frac: float = 0.70
    val_frac: float = 0.15

    use_huber: bool = True
    huber_beta: float = 1.0

    standardize_target_per_sensor: bool = False

    seed: int = 41
    device: str = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    )

    exclude_sensors: Tuple[str, ...] = ()
    loso: bool = False
    held_out_sensor: Optional[str] = None
    use_unk_for_heldout: bool = True
    unk_token: str = "__UNK__"

    use_local_pm_history: bool = True

    # Multi-scale sigmas
    sigmas_km: Tuple[float, ...] = (2.0, 4.0, 8.0)

    # Wind-anisotropic kernel params (simple defaults)
    sigma_perp_km: float = 2.0
    sigma_par_km: float = 4.0
    upwind_gate: bool = True
    gate_alpha: float = 1.5
    wind_dir_is_from: bool = True  # set False if your wdir encodes "to" direction already


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Utilities
# ----------------------------
def add_wind_sincos(df: pd.DataFrame, wdir_col: str = "wdir") -> pd.DataFrame:
    df = df.copy() # Converts wind direction into degrees in wdir_sin and wdir_cos so that wind direction
    # becomes a continuous representation and we don't have discontinuity between 0 and 359 degrees
    wdir = pd.to_numeric(df[wdir_col], errors="coerce")
    radians = np.deg2rad(wdir)
    df["wdir_sin"] = np.sin(radians)
    df["wdir_cos"] = np.cos(radians)
    return df


def time_sort(df: pd.DataFrame) -> pd.DataFrame: # Ensures timelocal is numeric, drops missing timestamps
    # and sorts by (sn, timelocal); this is critical so your sliding windows are truly chronological per sensor
    df = df.copy()
    df["timelocal"] = pd.to_numeric(df["timelocal"], errors="coerce")
    df = df.dropna(subset=["timelocal"])
    df["timelocal"] = df["timelocal"].astype("int64")
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
        X_list.append(feats[i - seq_len: i])
        y_list.append(target[i + horizon])

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else np.empty((0,), dtype=np.float32)
    return X, y


# ----------------------------
# Dataset (pooled + sensor id)
# ----------------------------
class PooledSequenceDataset(Dataset):
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
class PooledLSTMRegressor(nn.Module): # Pool windows from all sensors; onto one training set; 
    # Each sample includes (X_seq, sensor_idx, y)
    # The PooledLSTMRegressor uses LSTM over sequence to get temporal representation
    # A sensor embedding to let the model learn sensor-specific biases/offsets
    # Head predicts next PM2.5
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
        h_last = h_n[-1]
        e = self.sensor_emb(sensor_idx)
        z = torch.cat([h_last, e], dim=-1)
        return self.head(z).squeeze(-1)


# ----------------------------
# Evaluation helpers
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


# ----------------------------
# Optional: per-sensor target standardization
# ----------------------------
def compute_sensor_target_stats(per_sensor: Dict[str, pd.DataFrame], target_col: str, cfg: Config) -> Dict[str, Tuple[float, float]]:
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
    loss_fn = nn.SmoothL1Loss(beta=cfg.huber_beta) if cfg.use_huber else nn.MSELoss()

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
# Spatial helpers
# ----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_distance_matrix(sensor_meta: pd.DataFrame, sn_col="sn", lat_col="lat", lon_col="lon"):
    # Build sensor-to-sensor distances and multi-scale Gaussian kernels at sigmas (2, 4, 8 km)
    sns = sensor_meta[sn_col].tolist()
    lat = sensor_meta[lat_col].to_numpy()
    lon = sensor_meta[lon_col].to_numpy()

    n = len(sns)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i, :] = haversine_km(lat[i], lon[i], lat, lon).astype(np.float32)
    return sns, D


def gaussian_weights(D_km: np.ndarray, sigma_km: float, exclude_self=True):
    W = np.exp(-(D_km ** 2) / (2 * (sigma_km ** 2))).astype(np.float32)
    if exclude_self:
        np.fill_diagonal(W, 0.0)
    return W


def latlon_to_xy_km(lat, lon, lat0=None, lon0=None): # Converts lat/lon into local x/y kilometers
    """Equirectangular projection to local tangent plane (km)."""
    R = 6371.0
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    if lat0 is None: lat0 = float(np.nanmean(lat))
    if lon0 is None: lon0 = float(np.nanmean(lon))

    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    lat0r = np.deg2rad(lat0)
    lon0r = np.deg2rad(lon0)

    x = R * (lonr - lon0r) * np.cos(lat0r)
    y = R * (latr - lat0r)
    return x.astype(np.float32), y.astype(np.float32)


def add_kernel_neighbor_feature(
    df: pd.DataFrame,
    sensor_order: list,
    W: np.ndarray,
    time_col: str,
    value_col: str,
    out_col: str = "pm25_neighbor",
    out_wsum: str = "nbr_wsum",
    out_neff: str = "nbr_neff",
):
    df = df.copy()
    df[out_col] = np.nan
    df[out_wsum] = np.nan
    df[out_neff] = np.nan

    sn_to_i = {sn: i for i, sn in enumerate(sensor_order)}
    n = len(sensor_order)

    for t, g in df.groupby(time_col):
        v = np.full((n,), np.nan, dtype=np.float32)
        present_idx = []

        for sn, val in zip(g["sn"].tolist(), g[value_col].to_numpy()):
            if sn in sn_to_i and pd.notna(val):
                i = sn_to_i[sn]
                v[i] = float(val)
                present_idx.append(i)

        if len(present_idx) == 0:
            continue

        available = ~np.isnan(v)
        nbr = np.full_like(v, np.nan)
        wsum_vec = np.full_like(v, np.nan)
        neff_vec = np.full_like(v, np.nan)

        for i in present_idx:
            w = W[i].copy()
            w[~available] = 0.0

            denom = float(w.sum())
            if denom > 1e-8:
                num = float((w * v).sum())
                nbr[i] = num / denom
                wsum_vec[i] = denom

                w2 = float((w ** 2).sum())
                if w2 > 1e-12:
                    neff_vec[i] = (denom ** 2) / w2

        idx = g.index.to_numpy()
        df.loc[idx, out_col] = [nbr[sn_to_i[sn]] if sn in sn_to_i else np.nan for sn in g["sn"].tolist()]
        df.loc[idx, out_wsum] = [wsum_vec[sn_to_i[sn]] if sn in sn_to_i else np.nan for sn in g["sn"].tolist()]
        df.loc[idx, out_neff] = [neff_vec[sn_to_i[sn]] if sn in sn_to_i else np.nan for sn in g["sn"].tolist()]

    return df


def build_multi_scale_weights(D_km: np.ndarray, sigmas_km: List[float], exclude_self=True):
    Ws = {}
    for s in sigmas_km:
        Ws[float(s)] = gaussian_weights(D_km, sigma_km=float(s), exclude_self=exclude_self)
    return Ws


def add_multi_scale_neighbor_features( # For each time t and at each sensor, we compute
        # the pm25_neighbor value; the Nadaraya-Watson smoothed PM 2.5 from other sensors at same time
        # nbr_wsum_s is sum of kernel weights, and nbr_neff_s is the neighbor count (# contributors)
    df: pd.DataFrame,
    sensor_order: list,
    Ws: Dict[float, np.ndarray],
    time_col: str,
    value_col: str,
    prefix: str = "pm25_neighbor",
):
    df_out = df.copy()
    for sigma in sorted(Ws.keys()):
        W = Ws[sigma]
        df_out = add_kernel_neighbor_feature(
            df=df_out,
            sensor_order=sensor_order,
            W=W,
            time_col=time_col,
            value_col=value_col,
            out_col=f"{prefix}_s{sigma:g}",
            out_wsum=f"nbr_wsum_s{sigma:g}",
            out_neff=f"nbr_neff_s{sigma:g}",
        )
    return df_out


def add_wind_aniso_neighbor_feature( # At each time t and at each sensor i, it computes
        # a wind-aligned anisotropic kernel
        #along distance = projection onto wind direction
        # perpendicular distance = crosswind distance
        # Weight decays will vary depending on direction of wind
    df: pd.DataFrame,
    sensor_order: list,
    sensor_xy: Dict[str, Tuple[float, float]],
    time_col: str,
    value_col: str,
    wspd_col: str = "wspd",
    wdir_sin_col: str = "wdir_sin",
    wdir_cos_col: str = "wdir_cos",
    sigma_perp_km: float = 2.0,
    sigma_par_km: float = 4.0,
    upwind_gate: bool = True,
    gate_alpha: float = 1.5,
    wind_dir_is_from: bool = True,
    out_col: str = "pm25_neighbor_wind",
    out_wsum: str = "nbr_wsum_wind",
    out_neff: str = "nbr_neff_wind",
    exclude_neighbor_sns: Optional[set] = None,   # LOSO leakage guard
):
    """
    Wind-aware anisotropic Nadaraya–Watson:
      along/perp computed in local (x,y) km plane.
    Note: This assumes your wdir_sin/cos are a usable wind unit vector direction basis.
    """
    df = df.copy()
    df[out_col] = np.nan
    df[out_wsum] = np.nan
    df[out_neff] = np.nan

    exclude_neighbor_sns = exclude_neighbor_sns or set()

    sn_to_i = {sn: i for i, sn in enumerate(sensor_order)}
    n = len(sensor_order)

    xs = np.full((n,), np.nan, dtype=np.float32)
    ys = np.full((n,), np.nan, dtype=np.float32)
    for sn, idx in sn_to_i.items():
        if sn in sensor_xy:
            xs[idx], ys[idx] = sensor_xy[sn]

    sig_perp2 = float(sigma_perp_km ** 2 + 1e-12)
    sig_par2 = float(sigma_par_km ** 2 + 1e-12)

    for t, g in df.groupby(time_col):
        v = np.full((n,), np.nan, dtype=np.float32)
        present_idx = []
        wind_sin = {}
        wind_cos = {}

        # build pm vector across sensors for this timestamp
        for sn, val, s, c in zip(
            g["sn"].tolist(),
            g[value_col].to_numpy(),
            g[wdir_sin_col].to_numpy() if wdir_sin_col in g.columns else np.full((len(g),), np.nan),
            g[wdir_cos_col].to_numpy() if wdir_cos_col in g.columns else np.full((len(g),), np.nan),
        ):
            if sn in exclude_neighbor_sns:
                continue
            if sn in sn_to_i and pd.notna(val):
                i = sn_to_i[sn]
                v[i] = float(val)
                present_idx.append(i)
                if pd.notna(s) and pd.notna(c):
                    wind_sin[i] = float(s)
                    wind_cos[i] = float(c)

        if len(present_idx) == 0:
            continue

        available = ~np.isnan(v)
        nbr = np.full_like(v, np.nan)
        wsum_vec = np.full_like(v, np.nan)
        neff_vec = np.full_like(v, np.nan)

        for i in present_idx:
            dx = xs - xs[i]
            dy = ys - ys[i]

            if i not in wind_sin or i not in wind_cos:
                # fallback isotropic in xy
                d2 = (dx * dx + dy * dy).astype(np.float32)
                w = np.exp(-d2 / (2.0 * sig_perp2)).astype(np.float32)
            else:
                # wind unit vector (x=cos, y=sin in the same basis as wdir_sin/cos)
                ux = wind_cos[i]
                uy = wind_sin[i]
                if wind_dir_is_from:
                    ux, uy = -ux, -uy  # "from" -> "to"

                along = dx * ux + dy * uy
                perp = np.abs(dx * uy - dy * ux)

                w = np.exp(
                    -0.5 * (perp * perp) / sig_perp2
                    -0.5 * (along * along) / sig_par2
                ).astype(np.float32)

                if upwind_gate:
                    gate = 1.0 / (1.0 + np.exp(gate_alpha * along))
                    w = (w * gate).astype(np.float32)

            w[i] = 0.0
            w[~available] = 0.0

            denom = float(w.sum())
            if denom > 1e-8:
                num = float((w * v).sum())
                nbr[i] = num / denom
                wsum_vec[i] = denom

                w2 = float((w ** 2).sum())
                if w2 > 1e-12:
                    neff_vec[i] = (denom ** 2) / w2

        idx = g.index.to_numpy()
        df.loc[idx, out_col] = [nbr[sn_to_i[sn]] if sn in sn_to_i else np.nan for sn in g["sn"].tolist()]
        df.loc[idx, out_wsum] = [wsum_vec[sn_to_i[sn]] if sn in sn_to_i else np.nan for sn in g["sn"].tolist()]
        df.loc[idx, out_neff] = [neff_vec[sn_to_i[sn]] if sn in sn_to_i else np.nan for sn in g["sn"].tolist()]

    return df


# ----------------------------
# Main: pooled training
# ----------------------------
def train_pooled_lstm_with_embedding(df: pd.DataFrame, cfg: Config):
    set_seed(cfg.seed)

    df = time_sort(df)
    df = add_wind_sincos(df)

    target_col = "pm25_mean"
    df["pm25_in"] = pd.to_numeric(df[target_col], errors="coerce")

    meta = (
        df[["sn", "lat", "lon"]]
        .dropna()
        .drop_duplicates(subset=["sn"])
        .sort_values("sn")
        .reset_index(drop=True)
    )

    sensor_order, D_km = build_distance_matrix(meta, sn_col="sn", lat_col="lat", lon_col="lon")

    # time column (must be set BEFORE spatial feature creation)
    df["timelocal"] = pd.to_numeric(df["timelocal"], errors="coerce")
    df = df.dropna(subset=["timelocal"])
    df["timelocal"] = df["timelocal"].astype("int64")
    time_col = "timelocal"

    # XY for wind-anisotropic weights
    x_km, y_km = latlon_to_xy_km(meta["lat"].to_numpy(), meta["lon"].to_numpy())
    sensor_xy = {sn: (float(x), float(y)) for sn, x, y in zip(meta["sn"].tolist(), x_km, y_km)}

    # LOSO leakage prevention for wind feature: held-out sensor cannot contribute as a neighbor
    exclude_wind_neighbors = set()
    if cfg.loso and cfg.held_out_sensor is not None:
        exclude_wind_neighbors.add(cfg.held_out_sensor)

    # Wind-anisotropic neighbor feature
    df = add_wind_aniso_neighbor_feature(
        df=df,
        sensor_order=sensor_order,
        sensor_xy=sensor_xy,
        time_col=time_col,
        value_col="pm25_in",
        sigma_perp_km=cfg.sigma_perp_km,
        sigma_par_km=cfg.sigma_par_km,
        upwind_gate=cfg.upwind_gate,
        gate_alpha=cfg.gate_alpha,
        wind_dir_is_from=cfg.wind_dir_is_from,
        out_col="pm25_neighbor_wind",
        out_wsum="nbr_wsum_wind",
        out_neff="nbr_neff_wind",
        exclude_neighbor_sns=exclude_wind_neighbors,
    )

    # Multi-scale isotropic neighbor features
    Ws = build_multi_scale_weights(D_km, list(cfg.sigmas_km), exclude_self=True)

    # LOSO leakage prevention: held-out sensor does not contribute to others in isotropic weights
    if cfg.loso and cfg.held_out_sensor is not None and cfg.held_out_sensor in sensor_order:
        j = sensor_order.index(cfg.held_out_sensor)
        for W in Ws.values():
            W[:, j] = 0.0
            W[j, j] = 0.0

    df = add_multi_scale_neighbor_features(
        df=df,
        sensor_order=sensor_order,
        Ws=Ws,
        time_col=time_col,
        value_col="pm25_in",
        prefix="pm25_neighbor",
    )

    nbr_cols = [f"pm25_neighbor_s{s:g}" for s in cfg.sigmas_km]
    df["pm25_neighbor_mean"] = df[nbr_cols].mean(axis=1)
    df["pm25_spatial_resid"] = df["pm25_in"] - df["pm25_neighbor_mean"]

    # --- PATCH: make wind features non-NaN by falling back to isotropic aggregates ---
    wsum_cols = [f"nbr_wsum_s{s:g}" for s in cfg.sigmas_km]
    neff_cols = [f"nbr_neff_s{s:g}" for s in cfg.sigmas_km]

    df["nbr_wsum_mean"] = df[wsum_cols].mean(axis=1)
    df["nbr_neff_mean"] = df[neff_cols].mean(axis=1)

    df["pm25_neighbor_wind"] = df["pm25_neighbor_wind"].fillna(df["pm25_neighbor_mean"])
    df["nbr_wsum_wind"]      = df["nbr_wsum_wind"].fillna(df["nbr_wsum_mean"])
    df["nbr_neff_wind"]      = df["nbr_neff_wind"].fillna(df["nbr_neff_mean"])
# ------------------------------------------------------------


    # Features (NOW includes wind anisotropic features)
    feature_cols = [ # Feature column; tampering with the pm25_neighbor inputs here 
        "prcp", "pres", "rhum", "temp",
        "wspd", "wdir_sin", "wdir_cos",
        "pm25_neighbor_wind", "nbr_wsum_wind", "nbr_neff_wind",
    ]

    for s in cfg.sigmas_km:
        feature_cols += [f"pm25_neighbor_s{s:g}", f"nbr_wsum_s{s:g}", f"nbr_neff_s{s:g}"]

    if cfg.use_local_pm_history:
        feature_cols += ["pm25_in", "pm25_spatial_resid"]

    for c in feature_cols + [target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    excluded = set(cfg.exclude_sensors)
    heldout_sn = cfg.held_out_sensor if (cfg.loso and cfg.held_out_sensor is not None) else None

    train_per_sensor = {}
    heldout_sensor_df = None

    for sn, g in df.groupby("sn"):
        g = g.dropna(subset=feature_cols + [target_col]).copy().reset_index(drop=True)
        if len(g) < (cfg.seq_len + cfg.horizon + 50):
            continue

        if heldout_sn is not None and sn == heldout_sn:
            heldout_sensor_df = g
            continue

        if sn in excluded:
            continue

        train_per_sensor[sn] = g

    if not train_per_sensor:
        raise ValueError("No training sensors have enough data after cleaning (after exclusions).")
    if cfg.loso and heldout_sn is not None and heldout_sensor_df is None:
        raise ValueError(f"LOSO requested but held-out sensor {heldout_sn} has no usable data after cleaning.")

    train_sensors = sorted(train_per_sensor.keys())
    sn_to_idx = {sn: i for i, sn in enumerate(train_sensors)}
    idx_to_sn = {i: sn for sn, i in sn_to_idx.items()}

    if cfg.loso and cfg.use_unk_for_heldout:
        unk_idx = len(sn_to_idx)
        sn_to_idx[cfg.unk_token] = unk_idx
        idx_to_sn[unk_idx] = cfg.unk_token

    print(f"Excluded sensors: {sorted(list(excluded))}")
    print(f"Using sensors ({len(train_sensors)}): {train_sensors}")

    y_stats = None
    if cfg.standardize_target_per_sensor:
        y_stats = compute_sensor_target_stats(train_per_sensor, target_col, cfg)

    # Fit ONE global scaler on TRAIN rows only
    train_rows = []
    for sn, g in train_per_sensor.items():
        tr_sl, _, _ = split_by_time(len(g), cfg.train_frac, cfg.val_frac)
        train_rows.append(g.loc[tr_sl, feature_cols].to_numpy())
    train_rows = np.vstack(train_rows)
    scaler = StandardScaler().fit(train_rows)

    X_tr_list, s_tr_list, y_tr_list = [], [], []
    X_va_list, s_va_list, y_va_list = [], [], []
    X_te_list, s_te_list, y_te_list = [], [], []

    for sn, g in train_per_sensor.items():
        tr_sl, va_sl, te_sl = split_by_time(len(g), cfg.train_frac, cfg.val_frac)

        g_scaled = g.copy()
        g_scaled[feature_cols] = scaler.transform(g_scaled[feature_cols].to_numpy())

        X, y = make_windows(g_scaled, feature_cols, target_col, cfg.seq_len, cfg.horizon)
        pred_indices = np.arange(len(y)) + cfg.seq_len

        train_mask = (pred_indices >= tr_sl.start) & (pred_indices < tr_sl.stop)
        val_mask   = (pred_indices >= va_sl.start) & (pred_indices < va_sl.stop)
        test_mask  = (pred_indices >= te_sl.start) & (pred_indices < te_sl.stop)

        if cfg.standardize_target_per_sensor:
            mu, sd = y_stats[sn]
            y = standardize_y(y, mu, sd)

        s_idx = sn_to_idx[sn]
        X_tr_list.append(X[train_mask]); y_tr_list.append(y[train_mask]); s_tr_list.append(np.full(train_mask.sum(), s_idx, np.int64))
        X_va_list.append(X[val_mask]);   y_va_list.append(y[val_mask]);   s_va_list.append(np.full(val_mask.sum(), s_idx, np.int64))
        X_te_list.append(X[test_mask]);  y_te_list.append(y[test_mask]);  s_te_list.append(np.full(test_mask.sum(), s_idx, np.int64))

    X_train = np.concatenate(X_tr_list, axis=0); y_train = np.concatenate(y_tr_list, axis=0); s_train = np.concatenate(s_tr_list, axis=0)
    X_val   = np.concatenate(X_va_list, axis=0); y_val   = np.concatenate(y_va_list, axis=0); s_val   = np.concatenate(s_va_list, axis=0)
    X_test  = np.concatenate(X_te_list, axis=0); y_test  = np.concatenate(y_te_list, axis=0); s_test  = np.concatenate(s_te_list, axis=0)

    train_loader = DataLoader(PooledSequenceDataset(X_train, s_train, y_train), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(PooledSequenceDataset(X_val,   s_val,   y_val),   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(PooledSequenceDataset(X_test,  s_test,  y_test),  batch_size=cfg.batch_size, shuffle=False)

    holdout_loader = None
    holdout_metrics = None

    if cfg.loso and cfg.held_out_sensor is not None:
        g = heldout_sensor_df.copy()
        g_scaled = g.copy()
        g_scaled[feature_cols] = scaler.transform(g_scaled[feature_cols].to_numpy())

        Xh, yh = make_windows(g_scaled, feature_cols, target_col, cfg.seq_len, cfg.horizon)
        pred_indices = np.arange(len(yh)) + cfg.seq_len
        _, _, te_sl = split_by_time(len(g_scaled), cfg.train_frac, cfg.val_frac)
        test_mask = (pred_indices >= te_sl.start) & (pred_indices < te_sl.stop)

        Xh_test = Xh[test_mask]
        yh_test = yh[test_mask]

        if len(yh_test) > 0:
            sh_test = np.full((len(yh_test),), sn_to_idx[cfg.unk_token], dtype=np.int64)
            holdout_loader = DataLoader(PooledSequenceDataset(Xh_test, sh_test, yh_test), batch_size=cfg.batch_size, shuffle=False)

    n_embeddings = len(train_sensors) + (1 if (cfg.loso and cfg.use_unk_for_heldout) else 0)

    model = PooledLSTMRegressor(
        n_features=len(feature_cols),
        n_sensors=n_embeddings,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        embed_dim=cfg.embed_dim,
    )

    print(
        f"\n=== POOLED TRAINING | sensors={len(train_sensors)} "
        f"| train windows={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"| lr={cfg.lr} embed_dim={cfg.embed_dim} huber={cfg.use_huber} "
        f"| sigmas_km={cfg.sigmas_km} wind=(perp={cfg.sigma_perp_km},par={cfg.sigma_par_km}) "
        f"| use_local_pm_history={cfg.use_local_pm_history} ==="
    )

    model = train_one_model(model, train_loader, val_loader, cfg)

    if holdout_loader is not None:
        holdout_metrics = eval_overall(model, holdout_loader, cfg.device)
        print(f"\n[LOSO HOLDOUT={cfg.held_out_sensor}] TEST RMSE={holdout_metrics['rmse']:.4f} | MAE={holdout_metrics['mae']:.4f}")

    test_overall = eval_overall(model, test_loader, cfg.device)

    return {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "feature_cols": feature_cols,
        "sn_to_idx": sn_to_idx,
        "idx_to_sn": idx_to_sn,
        "metrics_test_overall": test_overall,
        "holdout_metrics": holdout_metrics,
        "held_out_sensor": cfg.held_out_sensor,
        "cfg": cfg,
    }


# ----------------------------
# LOSO CV: supports one sigma-tuple or list of sigma-tuples
# ----------------------------
def run_loso_cv(df: pd.DataFrame, base_cfg: Config, sigmas=(2.0, 4.0, 8.0)):
    df = df.copy()
    sensors = sorted(df["sn"].dropna().unique().tolist())

    if len(sigmas) > 0 and isinstance(sigmas[0], (list, tuple)):
        sigma_sets = [tuple(map(float, s)) for s in sigmas]
    else:
        sigma_sets = [tuple(map(float, sigmas))]

    all_rows = []

    for sigma_set in sigma_sets:
        for held_out in sensors:
            print("\n" + "=" * 90)
            print(f"LOSO: sigmas_km={sigma_set} | held_out={held_out}")

            cfg = Config(**{**base_cfg.__dict__})
            cfg.sigmas_km = sigma_set
            cfg.loso = True
            cfg.held_out_sensor = held_out
            cfg.exclude_sensors = ()
            cfg.use_unk_for_heldout = True

            artifacts = train_pooled_lstm_with_embedding(df, cfg)
            hm = artifacts["holdout_metrics"]
            if hm is None:
                continue

            all_rows.append({
                "sigmas_km": sigma_set,
                "held_out": held_out,
                "rmse": hm["rmse"],
                "mae": hm["mae"],
            })

    summary = pd.DataFrame(all_rows)
    if len(summary):
        print("\n\n===== LOSO SUMMARY (per held-out sensor) =====")
        print(summary.sort_values(["sigmas_km", "rmse"], ascending=[True, False]).to_string(index=False))

        print("\n===== LOSO SUMMARY (mean over sensors) =====")
        agg = summary.groupby("sigmas_km")[["rmse", "mae"]].mean().reset_index()
        print(agg.sort_values("rmse").to_string(index=False))

    return summary


if __name__ == "__main__":
    df = pd.read_csv("/Users/gavinyu/Desktop/Air Quality Project/meteo_sensor_together_Joe.csv")

    base_cfg = Config(
        embed_dim=32,
        lr=3e-4,
        use_huber=False,
        standardize_target_per_sensor=False,
        use_local_pm_history=True,  # strict generalization
        sigmas_km=(2.0, 4.0, 8.0),

        # wind params (you can tune)
        sigma_perp_km=2.0,
        sigma_par_km=4.0,
        upwind_gate=True,
        gate_alpha=1.5,
        wind_dir_is_from=True,
    )

    run_loso_cv(df, base_cfg, sigmas=(2.0, 4.0, 8.0))
