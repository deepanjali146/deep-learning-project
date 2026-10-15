# RViT-D v2 — MTCNN-Improved Pipeline

This notebook trains and evaluates two models for deepfake detection:

- **ViT (no aug)** — a plain Vision Transformer (`vit_small_patch16_224`, ImageNet-pretrained),
  fine-tuned on FaceForensics++ with no perturbation augmentation.
- **RViT-D** — the same backbone, trained with **DeeperForensics-1.0's actual released
  perturbation code** (`EndlessSora/DeeperForensics-1.0`, `perturbation/distortions.py`)
  applied as training-time augmentation.

Both models are evaluated under an **intra-dataset** protocol (FaceForensics++ test split) and
a **cross-dataset** protocol (OpenForensics, 140k Real and Fake Faces).

## What's improved over the original notebook

- **MTCNN face cropping** instead of raw frames — tight, centered face crops extracted directly
  from the FaceForensics++ videos.
- **Leakage-checked splits** — no source video's frames appear in more than one split
  (train/val/test).
- **Differential learning rate** — the classifier head trains 10x faster than the ViT backbone.
- **Validation-set threshold calibration** (Youden's J statistic) instead of a fixed 0.5 cutoff,
  used when computing Accuracy/F1 at test time.
- **Early stopping, stochastic depth (drop path), label smoothing** for better generalization on
  a still-modest training set.

The core extension — applying DeeperForensics-1.0's official perturbation suite to a Vision
Transformer — is unchanged from the original notebook.

## Requirements

- Google Colab with a GPU runtime (T4 or better)
- A Google account with Drive access (all outputs are saved to Drive, never lost on disconnect)
- A Kaggle account + API token (for downloading FaceForensics++, OpenForensics, and 140k
  Real and Fake Faces)
- ~20 GB of free Colab local disk space during the FF++ download/extraction step

## How to run

Run cells **top to bottom** on a fresh runtime. Each expensive step is resumable and will skip
itself if its output already exists on Drive:

1. **Mount Drive + GPU check**
2. **Install dependencies** (`timm`, `opencv-python-headless`, `facenet-pytorch`, `kagglehub`)
3. **Clone DeeperForensics-1.0's official perturbation code**
4. **Configuration** — defines all paths, hyperparameters (`CFG`), and the current `RUN_NAME`
5. **Download FaceForensics++ (c23) videos** — ~17 GB, skipped if already downloaded
6. **Extract MTCNN face crops** — slow; resumable; skips videos already processed
7. **Copy crops from Drive to local disk** + preview
8. **Build balanced splits** (with leakage check)
9. **Download cross-dataset sources** (OpenForensics, 140k Real and Fake Faces)
10. **DeeperForensics-1.0 augmentation + Dataset/DataLoader** definitions
11. **Model, training loop, and evaluation helpers**
12. **Train the plain ViT baseline**
13. **Train RViT-D** (with DeeperForensics perturbation augmentation)
14. **Results table** — AUC, Accuracy, F1 across all evaluation settings
15. **Score-distribution histograms** (diagnostic)
16. **Robustness curves** — AUC vs. perturbation severity, per distortion type
17. **Final Drive listing**

### After a Colab disconnect/restart

Google Colab wipes all Python variables on a runtime restart, but **everything saved to Drive
persists**. After a restart, re-run every cell from the top in order — the download and
extraction cells will detect their outputs already exist on Drive and skip the expensive work,
so a full re-run after a restart is fast except for re-downloading any files that live only on
Colab's local disk (not Drive).

**Important:** running cells out of order, or re-running only one cell without its dependencies,
is the most common source of `NameError` / empty-list bugs in this notebook — if something
looks wrong (e.g., a list that should be non-empty is empty), check whether an earlier cell in
the chain was skipped or run stale.

## Outputs

Everything is saved under:

```
/content/drive/MyDrive/RViT-D/runs/<RUN_NAME>/
```

where `RUN_NAME` is set in the Configuration cell (default: `run_v4_mtcnn_improved`). Each run
folder contains:

- `config.json` — the exact `CFG` used for this run
- `vit_plain/` and `rvitd/` — each with `best.pt` (checkpoint), `meta.json` (best epoch,
  validation-tuned threshold), and `done.flag` (marks training complete, enables resume)
- `results.csv` — AUC / Accuracy / Balanced Accuracy / F1 / Real Recall / Fake Recall for every
  (model, test set) combination
- `robustness.csv` — AUC at each DeeperForensics-1.0 perturbation severity level (1-5), per
  distortion type, per model, per test set
- `score_hist.png` — logit-score distributions, real vs. fake, per model/test set
- `robustness.png` — robustness curves (AUC vs. severity) per perturbation type

A new `RUN_NAME` never overwrites an earlier run's folder, so previous results (including the
first notebook version's `run_1`, `run_2_ffpp`, `run_3_ffpp_balanced`, and
`run_deeperforensics_official_code` runs) remain on Drive untouched.

## Known limitations (see project report, Section 5)

- Training set size (~4,000 images/class) is much smaller than the full FaceForensics++
  training protocol used by established baselines (Xception, F3-Net), so absolute accuracy is
  not directly comparable to published numbers.
- DeeperForensics-1.0's perturbation suite targets real-world recording/compression artifacts,
  not GAN-generation artifacts — cross-dataset generalization to StyleGAN-based forgeries
  (140k Real and Fake Faces) is not expected to improve from this augmentation alone.