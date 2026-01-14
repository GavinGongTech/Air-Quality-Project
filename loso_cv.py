# loso_cv.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Metrics
# -----------------------------
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def corr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

# -----------------------------
# Config
# -----------------------------
@dataclass
class ExperimentConfig:
    sigma_km: float
    seed: int = 0
    lr: float = 3e-4
    embed_dim: int = 32
    use_sensor_embedding: bool = True
    unk_sensor_token: str = "__UNK__"
    # LOSO options
    loso: bool = False
    held_out_sensor: Optional[str] = None
    use_unk_for_heldout: bool = True
    unk_token: str = "__UNK__"

    # stricter generalization: remove local pm history inputs
    # (still allows neighbor pm25_neighbor computed from other sensors)
    use_local_pm_history: bool = True

# -----------------------------
# You must adapt these 3 functions to your codebase.
# They should already exist in some form in LSTM_with_kernel.py
# -----------------------------
def load_full_dataframe() -> pd.DataFrame:
    """
    Return a dataframe with at least columns:
      - 'sn' (sensor id string)
      - 't'  (timestamp)
      - 'y'  (target)
    plus any covariates you use.
    """
    raise NotImplementedError

def build_datasets(
    df: pd.DataFrame,
    sigma_km: float,
    excluded_sensors: List[str],
    seed: int,
    use_sensor_embedding: bool,
    unk_sensor_token: str,
) -> Tuple[object, object, object, Dict[str, int]]:
    """
    Build (train_ds, val_ds, test_ds, sensor_to_idx).
    CRITICAL: kernel/neighbor features MUST be recomputed after excluding sensors.

    sensor_to_idx should include unk_sensor_token if use_sensor_embedding=True
    so that a held-out sensor can map to UNK during evaluation.
    """
    raise NotImplementedError

def train_and_predict(
    train_ds,
    val_ds,
    test_ds,
    sensor_to_idx: Dict[str, int],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """
    Train model, return predictions on test_ds as a DataFrame with columns:
      - 'sn' (sensor id for each sample)
      - 'y_true'
      - 'y_pred'
    """
    raise NotImplementedError

# -----------------------------
# LOSO runner
# -----------------------------
def run_loso(
    df: pd.DataFrame,
    sensors: List[str],
    cfg: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - per_fold: one row per held-out sensor with rmse/mae/corr + counts
      - all_preds: concatenated predictions across folds
    """
    rows = []
    all_preds = []

    for held_out in sensors:
        excluded = [held_out]

        # Build datasets from remaining sensors ONLY (no leakage)
        train_ds, val_ds, test_ds, sensor_to_idx = build_datasets(
            df=df,
            sigma_km=cfg.sigma_km,
            excluded_sensors=excluded,
            seed=cfg.seed,
            use_sensor_embedding=cfg.use_sensor_embedding,
            unk_sensor_token=cfg.unk_sensor_token,
        )

        preds = train_and_predict(train_ds, val_ds, test_ds, sensor_to_idx, cfg)

        # Evaluate ONLY on held-out sensor
        fold = preds[preds["sn"] == held_out].copy()
        if len(fold) == 0:
            raise RuntimeError(
                f"No test samples found for held-out sensor {held_out}. "
                f"Make sure your test split includes it, and your preds include sn."
            )

        fold_rmse = rmse(fold["y_true"], fold["y_pred"])
        fold_mae = mae(fold["y_true"], fold["y_pred"])
        fold_corr = corr(fold["y_true"], fold["y_pred"])

        rows.append({
            "held_out_sn": held_out,
            "sigma_km": cfg.sigma_km,
            "seed": cfg.seed,
            "rmse": fold_rmse,
            "mae": fold_mae,
            "corr": fold_corr,
            "n": len(fold),
        })

        all_preds.append(fold)

        print(f"[LOSO] held_out={held_out} | RMSE={fold_rmse:.4f} MAE={fold_mae:.4f} corr={fold_corr:.4f} n={len(fold)}")

    per_fold = pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    all_preds = pd.concat(all_preds, ignore_index=True)

    # Overall (equal-weighted by sample)
    overall = {
        "sigma_km": cfg.sigma_km,
        "seed": cfg.seed,
        "overall_rmse": rmse(all_preds["y_true"], all_preds["y_pred"]),
        "overall_mae": mae(all_preds["y_true"], all_preds["y_pred"]),
        "overall_corr": corr(all_preds["y_true"], all_preds["y_pred"]),
        "n": len(all_preds),
    }
    print(f"[LOSO overall] RMSE={overall['overall_rmse']:.4f} MAE={overall['overall_mae']:.4f} corr={overall['overall_corr']:.4f} n={overall['n']}")

    return per_fold, all_preds


if __name__ == "__main__":
    # Example usage:
    df = load_full_dataframe()
    sensors = sorted(df["sn"].unique().tolist())

    for sigma in [0.5, 1.0, 2.0, 4.0, 8.0]:
        cfg = ExperimentConfig(sigma_km=sigma, seed=0, use_sensor_embedding=True)
        per_fold, preds = run_loso(df, sensors, cfg)
        print("\nPer-fold summary (worst first):")
        print(per_fold.head(12))
