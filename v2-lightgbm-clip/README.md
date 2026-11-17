# v2: Colab LightGBM submission + CLIP image-feature experiments

This folder holds the second iteration of the pricing model.

| File | What it is |
|---|---|
| `notebooks/lightgbm_tfidf_colab.ipynb` | Colab version of the TF-IDF + LightGBM text model (validation SMAPE **51.73 %**). This is the approach described in [`docs/approach_document.md`](../docs/approach_document.md); its predictions are in `../v1-text-lightgbm/output/test_out.csv`. |
| `model/price_prediction_lgbm_model.txt`, `model/tfidf_vectorizer.pkl` | Trained booster and fitted vectorizer from that notebook. |
| `image_clip_embed_cv.py` | Image-only route A: frozen **CLIP ViT-B/32** (open_clip, OpenAI weights) embeddings → **Ridge** on log-price with 5-fold CV, reports CV SMAPE and writes `output/test_out.csv`. |
| `image_clip_head_finetune_fast.py` | Image-only route B: frozen CLIP backbone + small MLP head trained with AMP for a few epochs. |
| `attach_image_paths.py` | Adds an `image_path` column pointing at locally downloaded images, so the scripts read from disk instead of the URL. |
| `src/io_images.py`, `src/smape.py` | URL → PIL loader with grey-image fallback; SMAPE in NumPy and PyTorch. |
| `output/test_out.csv` | Test-set predictions from the CLIP route (no CV score was saved for this run). |

The CLIP scripts were written to run on a 4 GB laptop GPU (RTX 3050 Ti): the backbone is frozen, so the image encoder runs once and only a linear / small head is trained. An earlier attempt with CLIP ViT-L/14 needed about 10 GB of VRAM.

## Run

From this folder, with the challenge CSVs in `../dataset/`:

```bash
pip install -r requirements.txt

# A) embeddings + Ridge (5-fold CV)
python image_clip_embed_cv.py --train_csv ../dataset/train.csv --test_csv ../dataset/test.csv \
  --out_csv output/test_out.csv --arch ViT-B-32 --pretrained openai \
  --batch 64 --img_size 224 --num_workers 4 --folds 5 --seed 42

# B) frozen-backbone head fine-tune
python image_clip_head_finetune_fast.py --train_csv ../dataset/train.csv --test_csv ../dataset/test.csv \
  --out_csv output/test_out.csv --arch ViT-B-32 --pretrained openai \
  --epochs 3 --batch 48 --img_size 224 --num_workers 4
```

If you have downloaded the images, pass `--images_root_train` / `--images_root_test` (or run `attach_image_paths.py`) to avoid fetching every URL. Failed downloads fall back to a grey image. Predictions are clipped to positive values. Only generic pretrained CLIP weights are used; no external price data.
