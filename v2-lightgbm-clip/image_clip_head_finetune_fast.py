import argparse, os, random, time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
from src.io_images import fetch_image_or_gray
from urllib.parse import urlparse
import os.path as osp

def _resolve_local_path(row, images_root):
    p = row.get("image_path", None)
    if isinstance(p, str) and osp.exists(p):
        return p
    if not images_root:
        return None
    sid = str(int(row["sample_id"])) if "sample_id" in row and not pd.isna(row["sample_id"]) else None
    url = row.get("image_link", None)
    exts = [".jpg", ".jpeg", ".png", ".webp"]
    cand = []
    if sid:
        cand += [osp.join(images_root, sid + ext) for ext in exts]
    if isinstance(url, str) and url:
        base = osp.basename(urlparse(url).path)
        if base:
            cand.append(osp.join(images_root, base))
    for c in cand:
        if osp.exists(c):
            return c
    return None

class ImgDataset(Dataset):
    def __init__(self, df, preprocess, img_size=224, is_train=False, images_root=None):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.img_size = img_size
        self.is_train = is_train
        self.images_root = images_root
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        local = _resolve_local_path(row, self.images_root)
        if local and osp.exists(local):
            from PIL import Image
            try:
                img = Image.open(local).convert("RGB")
            except Exception:
                img = fetch_image_or_gray(row.get("image_link",""), size=self.img_size)
        else:
            img = fetch_image_or_gray(row.get("image_link",""), size=self.img_size)
        px = self.preprocess(img)
        if self.is_train:
            return px, float(row["price"])
        return px, int(row["sample_id"])

class ImageRegressor(nn.Module):
    def __init__(self, clip_model, head_dim=512, dropout=0.15, freeze_backbone=True):
        super().__init__()
        self.backbone = clip_model.visual
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        feat_dim = getattr(self.backbone, "output_dim", None)
        if feat_dim is None:
            proj = getattr(self.backbone, "proj", None)
            feat_dim = proj.shape[-1] if (proj is not None and hasattr(proj, "shape")) else 768
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.softplus = nn.Softplus()
    def encode(self, x):
        z = self.backbone(x)
        if isinstance(z, (list, tuple)):
            z = z[0]
        if z.dim() == 3:
            z = z.mean(1)
        return z
    def forward(self, x):
        z = self.encode(x)
        out = self.softplus(self.head(z)).squeeze(-1) + 1e-6
        return out

def train_one_epoch(model, loader, opt, scaler, device):
    model.train()
    loss_fn = nn.L1Loss()
    running = 0.0
    for px, y in tqdm(loader, desc="Train", leave=False):
        px = px.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y  = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred = model(px)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.detach().item() * px.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    abs_err = 0.0; denom_sum = 0.0
    for px, y in tqdm(loader, desc="Valid", leave=False):
        px = px.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y  = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            pred = model(px).clamp_min(1e-6)
        diff = (pred - y).abs()
        denom = (pred.abs() + y.abs()).clamp_min(1e-8) / 2.0
        abs_err += (diff / denom).sum().item()
        denom_sum += px.size(0)
    return (abs_err / denom_sum) * 100.0

def main(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    trn = pd.read_csv(args.train_csv).dropna(subset=["image_link","price","sample_id"]).reset_index(drop=True)
    tst = pd.read_csv(args.test_csv).dropna(subset=["image_link","sample_id"]).reset_index(drop=True)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(args.arch, pretrained=args.pretrained, device=device)
    clip_model.eval()

    # Split: last 5% as val
    val_size = max(1, int(len(trn)*0.05))
    trn_df = trn.iloc[:-val_size].reset_index(drop=True)
    val_df = trn.iloc[-val_size:].reset_index(drop=True)

    train_ds = ImgDataset(trn_df, preprocess, img_size=args.img_size, is_train=True, images_root=args.images_root_train)
    val_ds   = ImgDataset(val_df, preprocess, img_size=args.img_size, is_train=True, images_root=args.images_root_train)
    test_ds  = ImgDataset(tst, preprocess, img_size=args.img_size, is_train=False, images_root=args.images_root_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=True, drop_last=False, persistent_workers=(args.num_workers>0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False, persistent_workers=(args.num_workers>0))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False, persistent_workers=(args.num_workers>0))

    model = ImageRegressor(clip_model, head_dim=384, dropout=0.15, freeze_backbone=True).to(device).to(memory_format=torch.channels_last)
    opt = AdamW(model.head.parameters(), lr=2e-4, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler()

    best = float("inf"); best_state = None
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, scaler, device)
        val_smape = validate(model, val_loader, device)
        print(f"[epoch {epoch}] train L1={tr_loss:.6f} | val SMAPE={val_smape:.3f}%")
        if val_smape < best:
            best = val_smape
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference
    model.eval()
    preds = []
    ids = []
    with torch.no_grad():
        for px, sid in tqdm(test_loader, desc="Predict"):
            px = px.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast():
                yhat = model(px).clamp_min(1e-6)
            preds.extend(yhat.detach().cpu().numpy().tolist())
            ids.extend(sid.numpy().tolist())

    sub = pd.DataFrame({"sample_id": ids, "price": np.asarray(preds, dtype=float)}).sort_values("sample_id")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    sub.to_csv(args.out_csv, index=False)
    print(f"Wrote predictions -> {args.out_csv} (rows={len(sub)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_csv", default="output/test_out.csv")
    ap.add_argument("--arch", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--images_root_train", default=None)
    ap.add_argument("--images_root_test", default=None)
    args = ap.parse_args()
    main(args)
