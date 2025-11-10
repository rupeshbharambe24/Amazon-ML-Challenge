# Faster Image Pricing (CLIP Embeddings + Ridge CV)

This repo gives you two **efficient** routes that match your notebook's approach (image-based)
but run **much faster**, especially on 4GB GPUs like your RTX 3050 Ti.

## Options

### A) **CLIP Embeddings + Ridge (5-fold CV) — *fastest & most stable***
- Frozen **CLIP ViT-B/32** encoder (tiny VRAM) extracts **image embeddings** once.
- Train a **Ridge regressor** on embeddings (log-price), with **5-fold CV** and **SMAPE**.
- Generates `output/test_out.csv` in the required format.

### B) **Frozen-Backbone Head Fine-Tune — *still light, fully in PyTorch***
- CLIP **backbone is frozen**; you train only a small MLP head for a few epochs.
- Mixed precision (AMP), channels_last, cudnn.benchmark for speed.
- Outputs the same `output/test_out.csv`.

> Both paths clamp outputs to **positive** values and sort by `sample_id` exactly like your notebook.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the challenge CSVs here:
```
dataset/train.csv
dataset/test.csv
```

Run **A) Embeddings + Ridge (recommended first)**

```bash
python image_clip_embed_cv.py   --train_csv dataset/train.csv   --test_csv dataset/test.csv   --out_csv output/test_out.csv   --arch ViT-B-32 --pretrained openai   --batch 64 --img_size 224 --num_workers 4 --folds 5 --seed 42
```

Or **B) Frozen-backbone head fine-tune**

```bash
python image_clip_head_finetune_fast.py   --train_csv dataset/train.csv   --test_csv dataset/test.csv   --out_csv output/test_out.csv   --arch ViT-B-32 --pretrained openai   --epochs 3 --batch 48 --img_size 224 --num_workers 4
```

### Notes
- **Speed:** On your Ryzen 7 + 3050 Ti 4GB, Option A can run in well under an hour; Option B in ~30–60 min (dataset/download speed matters).
- **Fair play:** No external price data used. Only generic pretrained CLIP for *features*.
- **Robust I/O:** If an image URL fails, we auto-fallback to a gray image (same as your notebook).

---

## Why this is faster than your current notebook
- Uses **smaller CLIP** (ViT-B/32) instead of **ViT-L/14** (your logs showed ~10 GB VRAM).
- Freezes the heavy backbone (no backprop), so training is basically **linear regression** on embeddings.
- Mixed precision + efficient DataLoader settings.
- Keeping the pipeline deterministic with `--seed` and a fixed split strategy for CV.

---

## Files
- `image_clip_embed_cv.py` – extract embeddings, 5-fold CV Ridge on log-price, SMAPE CV, predict test.
- `image_clip_head_finetune_fast.py` – frozen-backbone head-only fine-tune, predict test.
- `src/smape.py` – SMAPE metric (numpy & torch variants).
- `src/io_images.py` – resilient URL → PIL Image loader (with fallback).
