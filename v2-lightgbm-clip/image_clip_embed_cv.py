import argparse, os, math, random, os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
from src.smape import smape_np
from src.io_images import fetch_image_or_gray
from urllib.parse import urlparse

def _resolve_local_path(row, images_root):
    # Priority 1: explicit image_path column
    p = row.get("image_path", None)
    if isinstance(p, str) and osp.exists(p):
        return p
    if not images_root:
        return None
    sid = str(int(row["sample_id"])) if "sample_id" in row and not pd.isna(row["sample_id"]) else None
    url = row.get("image_link", None)
    # Try common candidates
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

class ImgDataset(Dataset):
    def __init__(self, df, preprocess, img_size=224, images_root=None):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.img_size = img_size
        self.images_root = images_root
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        # Prefer LOCAL path if available
        local = _resolve_local_path(row, self.images_root)
        if local and osp.exists(local):
            from PIL import Image
            try:
                img = Image.open(local).convert("RGB")
            except Exception:
                img = fetch_image_or_gray(row.get("image_link",""), size=self.img_size)
        else:
            img = fetch_image_or_gray(row.get("image_link",""), size=self.img_size)
        return self.preprocess(img), row.get("price", np.nan), int(row["sample_id"])

@torch.no_grad()
def compute_embeddings(df, model, preprocess, device, batch=64, num_workers=4, img_size=224, images_root=None):
    ds = ImgDataset(df, preprocess, img_size=img_size, images_root=images_root)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last=False, persistent_workers=(num_workers>0))
    embs = []
    for px, _, _ in tqdm(loader, desc="Embed"):
        px = px.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast():
            z = model.encode_image(px)
        # z: [B, D]; ensure tensor
        if z.dim() == 3:
            z = z.mean(1)
        z = torch.nn.functional.normalize(z, dim=-1)  # optional
        embs.append(z.float().cpu().numpy())
    embs = np.concatenate(embs, axis=0)
    return embs

def make_folds_stratified_by_log_price(y, n_splits=5, seed=42):
    y_log = np.log1p(y)
    bins = np.floor(pd.qcut(y_log, q=20, labels=False, duplicates="drop")).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y_log), bins))

def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # Load CSVs
    trn = pd.read_csv(args.train_csv).dropna(subset=["sample_id","image_link","price"]).reset_index(drop=True)
    tst = pd.read_csv(args.test_csv).dropna(subset=["sample_id","image_link"]).reset_index(drop=True)

    # Model + preprocess
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.arch, pretrained=args.pretrained, device=device
    )
    clip_model.eval()

    # Compute embeddings
    train_emb = compute_embeddings(trn, clip_model, preprocess, device, images_root=args.images_root_train,
                                   batch=args.batch, num_workers=args.num_workers, img_size=args.img_size)
    test_emb  = compute_embeddings(tst, clip_model, preprocess, device, images_root=args.images_root_test,
                                   batch=args.batch, num_workers=args.num_workers, img_size=args.img_size)

    y = trn["price"].astype(float).values
    y_log = np.log1p(y)

    folds = make_folds_stratified_by_log_price(y, n_splits=args.folds, seed=args.seed)
    oof = np.zeros(len(trn), dtype=float)

    # Fit ridge on standardized embeddings
    scaler = StandardScaler(with_mean=False)  # sparse-friendly style; embeddings already dense but we keep it simple
    X_all = scaler.fit_transform(train_emb)
    Xt_all = scaler.transform(test_emb)

    for k, (tr_idx, va_idx) in enumerate(folds, 1):
        Xtr, Xva = X_all[tr_idx], X_all[va_idx]
        ytr = y_log[tr_idx]
        mdl = Ridge(alpha=2.0, random_state=args.seed, solver="auto")
        mdl.fit(Xtr, ytr)
        oof[va_idx] = mdl.predict(Xva)

    cv_smape = smape_np(y, np.expm1(oof))
    print(f"CV SMAPE: {cv_smape:.3f}% ({args.folds}-fold)")

    # Refit on full
    mdl_full = Ridge(alpha=2.0, random_state=args.seed, solver="auto")
    mdl_full.fit(X_all, y_log)
    yhat_test = np.expm1(mdl_full.predict(Xt_all)).clip(1e-6, None)

    sub = pd.DataFrame({"sample_id": tst["sample_id"].astype(int), "price": yhat_test.astype(float)})
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    sub.sort_values("sample_id").to_csv(args.out_csv, index=False)
    print(f"Wrote predictions -> {args.out_csv} (rows={len(sub)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_csv", default="output/test_out.csv")
    ap.add_argument("--arch", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--images_root_train", default=None)
    ap.add_argument("--images_root_test", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
