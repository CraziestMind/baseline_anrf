from models.baseline_model import FNO2D
from src.utils.config import load_config
from src.utils.utilities3 import *

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import os
from tqdm import tqdm

# -----------------------
# Load config
# -----------------------
cfg = load_config("configs/infer.yaml")

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Load normalization stats
# (computed from training data, not the bad .mat file)
# -----------------------
stats     = np.load(cfg.paths.norm_stats, allow_pickle=True).item()
min_vals  = stats['min']
max_vals  = stats['max']

min_pm = min_vals['cpm25']
max_pm = max_vals['cpm25']

def denorm(x):
    return x * (max_pm - min_pm) + min_pm

# -----------------------
# Settings
# -----------------------
time_input = cfg.data.time_input
time_out   = cfg.data.time_out
S1         = cfg.data.S1
S2         = cfg.data.S2

met_variables      = cfg.features.met_variables
emission_variables = cfg.features.emission_variables
all_features       = met_variables + emission_variables
V                  = len(all_features)

print(f"Features ({V}): {all_features}")

# -----------------------
# Dataset
# -----------------------
WIND_FEATURES     = ['u10', 'v10']
EMISSION_FEATURES = emission_variables  # all emissions including NMVOC_combined

class TestDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.arrs = {}
        for feat in all_features:
            if feat == 'NMVOC_combined':
                arr_e    = np.load(os.path.join(cfg.paths.input_loc, "NMVOC_e.npy"))
                arr_finn = np.load(os.path.join(cfg.paths.input_loc, "NMVOC_finn.npy"))
                self.arrs[feat] = (arr_e.astype(np.float32) + arr_finn.astype(np.float32)) / 2.0
                del arr_e, arr_finn
            else:
                self.arrs[feat] = np.load(
                    os.path.join(cfg.paths.input_loc, f"{feat}.npy")
                ).astype(np.float32)

        self.N = self.arrs[all_features[0]].shape[0]

    def __len__(self):
        return self.N

    def _normalize(self, arr, feat):
        lo  = min_vals[feat]
        hi  = max_vals[feat]
        den = np.where((hi - lo) == 0, 1.0, hi - lo)
        arr = (arr - lo) / den
        if feat in WIND_FEATURES:
            arr = 2.0 * arr - 1.0
        elif feat in EMISSION_FEATURES + ['NMVOC_combined']:
            arr = np.clip(arr, 0.0, 1.0)
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        x = np.empty((time_input, S1, S2, V), dtype=np.float32)
        for c, feat in enumerate(all_features):
            arr = self.arrs[feat][idx, :time_input]
            x[..., c] = self._normalize(arr, feat)
        return torch.from_numpy(x)


test_dataset = TestDataset()

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# =========================================================
# Model
# =========================================================
checkpoint = torch.load(cfg.paths.checkpoint, map_location=device)

model = FNO2D(
    time_in=time_input,
    features=V,
    time_out=time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded checkpoint: {cfg.paths.checkpoint}")

# =========================================================
# Inference
# =========================================================
os.makedirs(cfg.paths.output_loc, exist_ok=True)

prediction = np.zeros((len(test_dataset), S1, S2, time_out), dtype=np.float32)

with torch.no_grad():
    for i, x in enumerate(tqdm(test_loader)):
        x   = x.to(device, non_blocking=True)
        out = model(x).view(S1, S2, time_out)
        prediction[i] = out.cpu().numpy()

prediction = denorm(prediction)

out_path = os.path.join(cfg.paths.output_loc, 'preds.npy')
np.save(out_path, prediction)
print(f"Saved predictions → {out_path}  shape={prediction.shape}")
