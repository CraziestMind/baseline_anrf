import os
import numpy as np
from src.utils.config import load_config

cfg = load_config("configs/prepare_dataset.yaml")
RAW_PATH = cfg.paths.raw_path

WIND_FEATURES     = ['u10', 'v10']
MET_FEATURES      = ['cpm25', 'pblh', 'rain']
EMISSION_FEATURES = ['PM25', 'NH3', 'SO2', 'NOx']

SAVE_FEATURES    = MET_FEATURES + WIND_FEATURES + EMISSION_FEATURES + ['NMVOC_combined']
ALL_RAW_FEATURES = MET_FEATURES + WIND_FEATURES + EMISSION_FEATURES + ['NMVOC_e', 'NMVOC_finn']

def compute_gridwise_stats(months):
    print("\n=== Computing grid-wise normalization stats ===\n")
    min_vals = {}
    max_vals = {}

    for feat in ALL_RAW_FEATURES:
        print(f"  {feat}...")
        running_min = None
        running_max = None
        for month in months:
            arr = np.load(os.path.join(RAW_PATH, month, f"{feat}.npy")).astype(np.float32)
            m_min = arr.min(axis=0)
            m_max = arr.max(axis=0)
            del arr
            if running_min is None:
                running_min = m_min
                running_max = m_max
            else:
                running_min = np.minimum(running_min, m_min)
                running_max = np.maximum(running_max, m_max)
        min_vals[feat] = running_min
        max_vals[feat] = running_max

    min_vals['NMVOC_combined'] = np.minimum(min_vals['NMVOC_e'], min_vals['NMVOC_finn'])
    max_vals['NMVOC_combined'] = np.maximum(max_vals['NMVOC_e'], max_vals['NMVOC_finn'])
    return min_vals, max_vals

def normalize(arr, feat, min_vals, max_vals):
    lo  = min_vals[feat]
    hi  = max_vals[feat]
    den = np.where((hi - lo) == 0, 1.0, hi - lo)
    arr = (arr - lo) / den
    if feat in WIND_FEATURES:
        arr = 2.0 * arr - 1.0
    elif feat in EMISSION_FEATURES + ['NMVOC_combined']:
        arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)

def load_feature_month(feat, month, min_vals, max_vals):
    if feat == 'NMVOC_combined':
        arr_e    = np.load(os.path.join(RAW_PATH, month, "NMVOC_e.npy")).astype(np.float32)
        arr_finn = np.load(os.path.join(RAW_PATH, month, "NMVOC_finn.npy")).astype(np.float32)
        raw = (arr_e + arr_finn) / 2.0
        del arr_e, arr_finn
    else:
        raw = np.load(os.path.join(RAW_PATH, month, f"{feat}.npy")).astype(np.float32)
    return normalize(raw, feat, min_vals, max_vals)

def make_samples(arr, horizon, stride):
    return np.stack(
        [arr[i : i + horizon] for i in range(0, arr.shape[0] - horizon + 1, stride)],
        axis=0
    )

def train_val_split(samples, val_frac, seed):
    np.random.seed(seed)
    idx   = np.random.permutation(len(samples))
    n_val = int(val_frac * len(samples))
    return samples[idx[n_val:]], samples[idx[:n_val]]

os.makedirs(cfg.paths.train_savepath, exist_ok=True)
os.makedirs(cfg.paths.val_savepath,   exist_ok=True)

horizon  = cfg.data.horizon
stride   = cfg.data.stride
val_frac = cfg.data.val_frac
seed     = cfg.data.seed
months   = cfg.data.months

print(f"\n{'='*40}")
print(f"Train : {cfg.paths.train_savepath}")
print(f"Val   : {cfg.paths.val_savepath}")
print(f"Horizon={horizon}  Stride={stride}")
print(f"Features ({len(SAVE_FEATURES)}): {SAVE_FEATURES}")
print(f"{'='*40}\n")

# Compute stats
min_vals, max_vals = compute_gridwise_stats(months)
np.save(os.path.join(cfg.paths.train_savepath, "norm_stats.npy"),
        {'min': min_vals, 'max': max_vals})
np.save("/kaggle/working/norm_stats.npy",
        {'min': min_vals, 'max': max_vals})
print("Stats saved.\n")

# Save per-feature files (low memory, safe)
for feat in SAVE_FEATURES:
    print(f"\n=== {feat} ===")
    train_chunks, val_chunks = [], []

    for month in months:
        print(f"  [{month}]", end=" ")
        arr     = load_feature_month(feat, month, min_vals, max_vals)
        samples = make_samples(arr, horizon, stride)
        print(f"samples={len(samples)}", end=" ")
        train_s, val_s = train_val_split(samples, val_frac, seed)
        print(f"train={len(train_s)}  val={len(val_s)}")
        train_chunks.append(train_s)
        val_chunks.append(val_s)
        del arr, samples, train_s, val_s

    train_out = np.concatenate(train_chunks, axis=0)
    val_out   = np.concatenate(val_chunks,   axis=0)
    print(f"  → merged train={train_out.shape}  val={val_out.shape}")
    np.save(os.path.join(cfg.paths.train_savepath, f"train_{feat}.npy"),
            train_out.astype(np.float32))
    np.save(os.path.join(cfg.paths.val_savepath, f"val_{feat}.npy"),
            val_out.astype(np.float32))
    del train_chunks, val_chunks, train_out, val_out

print("\n=== Preparation complete ===")
