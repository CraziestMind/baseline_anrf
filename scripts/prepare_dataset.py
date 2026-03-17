import os
import numpy as np
from src.utils.config import load_config

# -----------------------
# Load config
# -----------------------
cfg = load_config("configs/prepare_dataset.yaml")
RAW_PATH = cfg.paths.raw_path

# -----------------------
# Feature definitions
# -----------------------
WIND_FEATURES     = ['u10', 'v10']
MET_FEATURES      = ['cpm25', 'pblh', 'rain']
EMISSION_FEATURES = ['PM25', 'NH3', 'SO2', 'NOx']

SAVE_FEATURES = MET_FEATURES + WIND_FEATURES + EMISSION_FEATURES + ['NMVOC_combined']
# = ['cpm25', 'pblh', 'rain', 'u10', 'v10', 'PM25', 'NH3', 'SO2', 'NOx', 'NMVOC_combined']

ALL_RAW_FEATURES = MET_FEATURES + WIND_FEATURES + EMISSION_FEATURES + ['NMVOC_e', 'NMVOC_finn']

# -----------------------
# Step 1: Compute grid-wise stats from all training months
# Grid-wise = per (lat, lon) point, shape (H, W)
# -----------------------
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
            m_min = arr.min(axis=0)  # (H, W)
            m_max = arr.max(axis=0)  # (H, W)
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

# -----------------------
# Step 2: Normalize
# -----------------------
def normalize(arr, feat, min_vals, max_vals):
    lo  = min_vals[feat]   # (H, W)
    hi  = max_vals[feat]   # (H, W)
    den = hi - lo
    den = np.where(den == 0, 1.0, den)  # avoid divide by zero at dead grid points

    arr = (arr - lo) / den              # broadcast over T dimension: (T, H, W)

    if feat in WIND_FEATURES:
        arr = 2.0 * arr - 1.0
    elif feat in EMISSION_FEATURES + ['NMVOC_combined']:
        arr = np.clip(arr, 0.0, 1.0)

    return arr.astype(np.float32)

# -----------------------
# Step 3: Sliding window
# -----------------------
def make_samples(arr, horizon, stride):
    return np.stack(
        [arr[i : i + horizon] for i in range(0, arr.shape[0] - horizon + 1, stride)],
        axis=0
    )  # (N, horizon, H, W)

# -----------------------
# Step 4: Train/val split — same indices for all features
# Returns indices, not data
# -----------------------
def get_split_indices(N, val_frac, seed):
    np.random.seed(seed)
    idx   = np.random.permutation(N)
    n_val = int(val_frac * N)
    return idx[n_val:], idx[:n_val]  # train_idx, val_idx

# -----------------------
# Step 5: Load and normalize one feature for one month
# -----------------------
def load_feature_month(feat, month, min_vals, max_vals):
    if feat == 'NMVOC_combined':
        arr_e    = np.load(os.path.join(RAW_PATH, month, "NMVOC_e.npy")).astype(np.float32)
        arr_finn = np.load(os.path.join(RAW_PATH, month, "NMVOC_finn.npy")).astype(np.float32)
        raw = (arr_e + arr_finn) / 2.0
        del arr_e, arr_finn
    else:
        raw = np.load(os.path.join(RAW_PATH, month, f"{feat}.npy")).astype(np.float32)

    return normalize(raw, feat, min_vals, max_vals)

# -----------------------
# Run
# -----------------------
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

# Compute grid-wise stats
min_vals, max_vals = compute_gridwise_stats(months)

# Save stats for inference (grid-wise, shape H x W per feature)
np.save(
    os.path.join(cfg.paths.train_savepath, "norm_stats.npy"),
    {'min': min_vals, 'max': max_vals}
)
# Backup to working dir (persists after session)
np.save("/kaggle/working/norm_stats.npy", {'min': min_vals, 'max': max_vals})
print("Stats saved.\n")

# Build stacked dataset month by month
# Final output: (N, horizon, H, W, F) — single array, one read per sample in dataloader
train_chunks = []
val_chunks   = []

for month in months:
    print(f"\n=== Month: {month} ===")

    # Load all features for this month: list of (T, H, W)
    feature_arrays = []
    for feat in SAVE_FEATURES:
        arr = load_feature_month(feat, month, min_vals, max_vals)
        feature_arrays.append(arr)
        print(f"  loaded {feat}: {arr.shape}")

    # Stack features: (T, H, W, F)
    stacked = np.stack(feature_arrays, axis=-1)
    del feature_arrays
    print(f"  stacked: {stacked.shape}")

    # Sliding window over time: (N, horizon, H, W, F)
    samples = np.stack(
        [stacked[i : i + horizon] for i in range(0, stacked.shape[0] - horizon + 1, stride)],
        axis=0
    )
    del stacked
    print(f"  samples: {samples.shape}")

    # Split indices (consistent across features since stacked)
    train_idx, val_idx = get_split_indices(len(samples), val_frac, seed)
    train_chunks.append(samples[train_idx])
    val_chunks.append(samples[val_idx])
    del samples

# Concatenate across months
train_data = np.concatenate(train_chunks, axis=0).astype(np.float32)
val_data   = np.concatenate(val_chunks,   axis=0).astype(np.float32)
del train_chunks, val_chunks

print(f"\nFinal train: {train_data.shape}  {train_data.nbytes / 1e9:.1f} GB")
print(f"Final val:   {val_data.shape}  {val_data.nbytes / 1e9:.1f} GB")

np.save(os.path.join(cfg.paths.train_savepath, "train_data.npy"), train_data)
np.save(os.path.join(cfg.paths.val_savepath,   "val_data.npy"),   val_data)
del train_data, val_data

print("\n=== Preparation complete ===")
