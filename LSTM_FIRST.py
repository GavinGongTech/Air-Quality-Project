# My first attempt at the LSTM architecture for this problem. 
# I ASSUME EACH SENSOR's DATA IS UNCORRELATED FOR NOW AMONG OTHER SENSORS. 

# Here is the general approach of my code
"""
1. Sort each sensor's data by time
2. Encode the wind direction as sin(wdir) and cos(wdir); this is important because the degrees are circular
3. IMPORTANT: Time Split; I will split the data into training, validation, and test sets based on time, 
So just do train = first chunk, val = next chunk, test = last chunk
4. Scale features using only the train split
5. Build the sliding windows; X[i] = features from times [i-L, ..., i-1] and Y[i] = PM2.5 at time i + h

So the LSTM takes the L-hour feature sequence, takes the last hidden state -> linear layer -> predicted PM2.5
And it trains the LSTM regressor with early stopping using mean squared error and reports root mean squared error 
"""

import numpy as np # Standard libraries for our usage
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict

from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Config: centralizes hyperparameters

@dataclass
class Config:
    """
    Stores all hyperparameters in one place so experiments are reproducible

    seq_len: number of past hours to feed into the model
    horizon: forecast horizon; 1 means predict PM2.5 one hour ahead
    """
    seq_len: int = 24
    horizon: int = 1
    batch_size: int = 64
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    train_frac: float = 0.70
    val_frac: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # device auto-select: uses GPU if available for speed; otherwise CPU.

# Utilities: feature engineering, sorting, splitting, windowing
def add_wind_sincos(df: pd.DataFrame, wdir_col: str = "wdir") -> pd.DataFrame: # Step 2.
    """
    Wind direction is circular:
      0° and 360° are the same direction.
    If we model raw degrees, the model sees a false discontinuity between 0 and 359 degrees. 

    Therefore, we represent direction by sin/cos on the unit circle.
    """
    df = df.copy()  # avoids modifying original dataframe in-place

    # Convert to numeric safely; invalid values -> NaN
    wdir = pd.to_numeric(df[wdir_col], errors="coerce")

    # Convert degrees to radians for np.sin/np.cos
    radians = np.deg2rad(wdir)

    # New engineered features
    df["wdir_sin"] = np.sin(radians)
    df["wdir_cos"] = np.cos(radians)

    return df


def time_sort(df: pd.DataFrame) -> pd.DataFrame: # Step 1; sort the data
    """
    Sorts data chronologically *within each sensor*.

    Why needed:
    - Window construction assumes the data is in correct time order.
    - Avoids training on mixed-up time steps which breaks sequence models.
    """
    df = df.copy()

    # Try to parse "time" as datetime if present
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # If parsing worked for at least some rows, sort by (sn, time)
    if "time" in df.columns and df["time"].notna().any():
        df = df.sort_values(["sn", "time"])
    else:
        # Otherwise fall back to timelocal
        df["timelocal"] = pd.to_numeric(df["timelocal"], errors="coerce")
        df = df.sort_values(["sn", "timelocal"])

    # Reset index for clean downstream slicing and window indexing
    return df.reset_index(drop=True)


def split_by_time(n: int, train_frac: float, val_frac: float) -> Tuple[slice, slice, slice]:# Step 3
    """
    Produces slices that partition the dataset into train/val/test

    Why needed:
    - Random splits cause data leakage in time series (future informs past).
    - Time-based split mimics real forecasting; ALWAYS make sure we don't have leakage
    """
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


class SequenceDataset(Dataset): # Just simple formatting so that we can insert into a dataloader
    """
    Wraps windowed arrays into a PyTorch Dataset so we can use DataLoader.
    X shape: (N, seq_len, n_features)
    y shape: (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Convert numpy arrays to torch tensors once (efficient)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]  # number of training examples/windows

    def __getitem__(self, idx):
        # returns one (sequence, target) pair
        return self.X[idx], self.y[idx]


def make_windows( # Step 5; create the sliding windows
    df_sensor: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a sensor's time series into supervised learning samples.

    Example:
      seq_len=24, horizon=1
      X uses hours [t-24, ..., t-1]
      y is PM2.5 at hour [t+1]

    Why needed:
    - LSTM expects sequences (T timesteps each).
    - Sliding windows create many training samples from one series.
    """
    # Extract features matrix (T, F)
    feats = df_sensor[feature_cols].to_numpy(dtype=np.float32)

    # Extract target vector (T,)
    target = pd.to_numeric(df_sensor[target_col], errors="coerce").to_numpy(dtype=np.float32)

    # Remove rows where target is NaN so we can train supervised regression
    # (Also aligns feats and target by applying the same mask)
    valid = ~np.isnan(target)
    feats = feats[valid]
    target = target[valid]

    N = len(target)
    X_list, y_list = [], []

    # i is the "current time index" for the window end
    # Window uses [i-seq_len, ..., i-1], prediction at [i+horizon]
    for i in range(seq_len, N - horizon):
        X_list.append(feats[i - seq_len : i])  # shape (seq_len, F)
        y_list.append(target[i + horizon])     # scalar

    # Stack into final arrays:
    # X: (num_windows, seq_len, F)
    # y: (num_windows,)
    X = np.stack(X_list, axis=0) if X_list else np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else np.empty((0,), dtype=np.float32)
    return X, y



# Model: LSTM -> last hidden state -> linear regression head; the heart of our model here

class LSTMRegressor(nn.Module):
    """
    Reads a sequence of feature vectors and outputs one PM2.5 prediction.

    Architecture:
    LSTM encoder -> last hidden state -> Dropout -> Linear -> y_hat
    """
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()

        # LSTM core. batch_first=True => input shape (B, T, F)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # dropout only applies between LSTM layers (so must have num_layers>1)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # "Head" maps hidden state to scalar prediction
        self.head = nn.Sequential(
            nn.Dropout(dropout),     # regularization to reduce overfitting
            nn.Linear(hidden_size, 1)  # regression output
        )

    def forward(self, x):
        """
        x: (B, T, F)

        Returns:
          y_hat: (B,)
        """
        # out: (B, T, hidden_size)
        # h_n: (num_layers, B, hidden_size) = hidden state at final timestep per layer
        out, (h_n, c_n) = self.lstm(x)

        # Use last layer's hidden state at final timestep as sequence summary
        h_last = h_n[-1]  # (B, hidden_size)

        # Linear regressor head
        y_hat = self.head(h_last).squeeze(-1)  # (B,)
        return y_hat

# Evaluation & Training

@torch.no_grad()
def eval_metrics(model, loader, device) -> Dict[str, float]:
    """
    Computes RMSE and MAE on a dataset loader.
    @torch.no_grad disables gradient tracking -> faster and less memory.
    """
    model.eval()
    ys, preds = [], []

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        pb = model(Xb)

        # move to CPU + numpy for metric computations
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
    """
    Trains one model using early stopping on validation RMSE.

    Why early stopping:
    - Helps prevent overfitting since the data for a neural network here is limited
    - Picks the checkpoint that generalizes best to unseen times.
    """
    device = cfg.device
    model = model.to(device)

    # AdamW = Adam with decoupled weight decay; stable for deep nets (inclusion of hidden layers in network)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # MSE is standard for regression; we report RMSE for interpretability
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)  # clears old gradients efficiently
            pred = model(Xb)
            loss = loss_fn(pred, yb)

            loss.backward()  # compute gradients

            # Gradient clipping helps prevent exploding gradients in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()  # update weights

        # Evaluate on validation set each epoch
        val = eval_metrics(model, val_loader, device)
        val_rmse = val["rmse"]

        print(f"Epoch {epoch:03d} | val RMSE={val_rmse:.4f} | val MAE={val['mae']:.4f}")

        # Track best validation performance
        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            # Save a CPU copy of the best model weights
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print("Early stopping.")
                break

    # Restore best checkpoint before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return model



# Main: train per sensor; Approach 1; train per sensor

def train_per_sensor_lstm(df: pd.DataFrame, cfg: Config):
    """
    Trains a separate LSTM model per sensor.

    Why per-sensor:
    - Useful for exploratory analysis: which sensors are predictable?
    - Detects sensor-specific noise, calibration, or missingness issues.

    Caveat:
    - Each sensor has limited data; pooled model often generalizes better and will represent my next approach
    """
    df = time_sort(df)
    df = add_wind_sincos(df)

    # Feature selection: meteorology + wind sin/cos
    feature_cols = [
        "prcp", "pres", "rhum", "temp",
        "wspd",
        "wdir_sin", "wdir_cos",
    ]
    target_col = "pm25_mean"

    # Enforce numeric types (strings -> floats, invalid -> NaN)
    for c in feature_cols + [target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    results = {}

    # group by sn => train independently per sensor
    for sn, g in df.groupby("sn"): # These are the six steps put together in the pipeline
        # Drop any row with missing input feature or missing target
        g = g.dropna(subset=feature_cols + [target_col]).copy()
        g = g.reset_index(drop=True)

        # sanity check: ensure enough rows to form windows
        if len(g) < (cfg.seq_len + cfg.horizon + 50):
            print(f"[{sn}] Skipping (not enough data after cleaning): {len(g)} rows")
            continue

        # Time-based split prevents leakage from future into past
        tr_sl, va_sl, te_sl = split_by_time(len(g), cfg.train_frac, cfg.val_frac)

        # Scale features: fit ONLY on train set to avoid leakage
        scaler = StandardScaler()
        scaler.fit(g.loc[tr_sl, feature_cols].to_numpy())

        # Apply scaling to all rows so model sees normalized inputs
        g_scaled = g.copy()
        g_scaled[feature_cols] = scaler.transform(g_scaled[feature_cols].to_numpy())

        # Construct windows over the *whole* series
        X, y = make_windows(g_scaled, feature_cols, target_col, cfg.seq_len, cfg.horizon)

        # Each y corresponds to an original time index i = k + seq_len
        pred_indices = np.arange(len(y)) + cfg.seq_len

        # Assign windows to train/val/test based on prediction timestamp
        train_mask = (pred_indices >= tr_sl.start) & (pred_indices < tr_sl.stop)
        val_mask   = (pred_indices >= va_sl.start) & (pred_indices < va_sl.stop)
        test_mask  = (pred_indices >= te_sl.start) & (pred_indices < te_sl.stop)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val     = X[val_mask], y[val_mask]
        X_test, y_test   = X[test_mask], y[test_mask]

        # Wrap as PyTorch datasets
        train_ds = SequenceDataset(X_train, y_train)
        val_ds   = SequenceDataset(X_val, y_val)
        test_ds  = SequenceDataset(X_test, y_test)

        # DataLoaders batch examples for training; shuffle only train
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

        # Instantiate model
        model = LSTMRegressor(
            n_features=len(feature_cols),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

        print(f"\n=== Training sensor {sn} | train windows={len(train_ds)} val={len(val_ds)} test={len(test_ds)} ===")
        model = train_one_model(model, train_loader, val_loader, cfg)

        # Final evaluation on held-out test segment
        test = eval_metrics(model, test_loader, cfg.device)
        print(f"[{sn}] TEST RMSE={test['rmse']:.4f} | TEST MAE={test['mae']:.4f}")

        # Save artifacts needed to reproduce inference later:
        # - trained weights
        # - scaler used to normalize features
        # - feature column ordering (must match at inference)
        results[sn] = {
            "test_rmse": test["rmse"],
            "test_mae": test["mae"],
            "scaler": scaler,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "feature_cols": feature_cols,
        }

    return results

df = pd.read_csv("/Users/gavinyu/Desktop/Air Quality Project/meteo_sensor_together_Joe.csv")
cfg = Config(seq_len=24, horizon=1, hidden_size=64, num_layers=2)
results = train_per_sensor_lstm(df, cfg)
