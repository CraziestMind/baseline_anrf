from src.utils.utilities3 import *
from src.utils.adam import Adam
from models.baseline_model import FNO2D
from src.utils.config import load_config
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

import torch
torch.set_num_threads(1)

import numpy as np
import json
import os
import time
from tqdm import tqdm

# -----------------------
# Load config
# -----------------------
cfg = load_config("configs/train.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.cuda.empty_cache()

torch.manual_seed(0)
np.random.seed(0)

# -----------------------
# Settings
# -----------------------
time_input = cfg.data.time_input
time_out   = cfg.data.time_out
T          = time_input + time_out
S1         = cfg.data.S1
S2         = cfg.data.S2

met_variables      = cfg.features.met_variables
emission_variables = cfg.features.emission_variables
all_features       = met_variables + emission_variables
V                  = len(all_features)

batch_size     = cfg.training.batch_size
epochs         = cfg.training.epochs
savepath_train = cfg.paths.savepath_train
savepath_val   = cfg.paths.savepath_val

print(f"Features ({V}): {all_features}")

# =========================================================
# Dataset — full preload into RAM
# Loads all per-feature files and stacks into one array
# (N, T, H, W, F) fully in RAM — zero file I/O during training
# =========================================================
class DataLoaders(torch.utils.data.Dataset):

    def __init__(self, split):
        base_path = savepath_train if split == "train" else savepath_val
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split: {split}")

        print(f"\nPreloading {split} dataset into RAM...")
        N = None
        self.data = None

        for i, feat in enumerate(all_features):
            path = os.path.join(base_path, f"{split}_{feat}.npy")
            arr  = np.load(path).astype(np.float32)  # (N, 26, H, W)

            if self.data is None:
                N = arr.shape[0]
                self.data = np.empty((N, T, S1, S2, V), dtype=np.float32)

            self.data[..., i] = arr[:, :T]
            del arr
            print(f"  loaded {feat} ({i+1}/{V})")

        self.N = N
        print(f"  Done. shape={self.data.shape}  "
              f"RAM={self.data.nbytes/1e9:.1f} GB\n")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample = self.data[idx]                                    # (T, H, W, F)
        x = torch.from_numpy(sample[:time_input].copy())          # (10, H, W, F)
        y = torch.from_numpy(
                sample[time_input:, ..., 0].copy()                # (16, H, W)
            ).permute(1, 2, 0)                                     # (H, W, 16)
        return x, y


train_dataset = DataLoaders("train")
val_dataset   = DataLoaders("val")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,      # 0 because data is already in RAM, no I/O needed
    pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# =========================================================
# Model
# =========================================================
model = FNO2D(
    time_in=time_input,
    features=V,
    time_out=time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
).to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f"Total parameters: {count_params(model)}")

optimizer = Adam(
    model.parameters(),
    lr=float(cfg.training.lr),
    weight_decay=float(cfg.training.weight_decay)
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.training.scheduler_step,
    gamma=cfg.training.scheduler_gamma
)

myloss = LpLoss(size_average=False)

os.makedirs(os.path.dirname(cfg.paths.model_save_path), exist_ok=True)
os.makedirs(os.path.dirname(cfg.paths.save_dir),        exist_ok=True)

log = []

# =========================================================
# Training loop
# =========================================================
for ep in tqdm(range(epochs)):
    model.train()
    t_start  = time.time()
    train_l2 = 0.0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x).view(x.size(0), S1, S2, time_out)
        l2  = myloss(out, y)
        l2.backward()
        optimizer.step()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    val_l2 = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x).view(x.size(0), S1, S2, time_out)
            val_l2 += myloss(out, y).item()

    train_l2 /= len(train_dataset)
    val_l2   /= len(val_dataset)
    duration  = time.time() - t_start

    log.append({
        "epoch":    ep,
        "duration": duration,
        "train_l2": train_l2,
        "val_l2":   val_l2,
    })

    print(f"ep={ep}  t={duration:.1f}s  train={train_l2:.4f}  val={val_l2:.4f}")

    if (ep + 1) % cfg.training.checkpoint_every == 0:
        ckpt_path = cfg.paths.model_save_path.replace(".pt", f"_ep{ep}.pt")
        torch.save({
            'epoch':                ep,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, ckpt_path)

        with open(cfg.paths.save_dir, "w") as f:
            json.dump(log, f)

print("\n=== Training complete ===")
