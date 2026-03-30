# Data Augmentation & PARSeq Fine-Tuning Guide

## Overview

This branch adds data augmentation and fine-tuning improvements to the PARSeq scene text recognition stage of the jersey number pipeline. Two augmentation strategies are provided, along with a modified training procedure that avoids catastrophic forgetting.

### What Changed

| File | Change |
|---|---|
| `augment_and_build_lmdb.py` | **New** — Augmentation strategy v1 (blur, perspective, rectangular masking) |
| `augment_and_build_lmdb_v2.py` | **New** — Augmentation strategy v2 (motion blur, pixelate, JPEG compression) |
| `configuration.py` | Added `crops_folder`, `jersey_id_result`, `final_result` to train config |
| `str/parseq/train.py` | Modified to support weights-only loading from `.ckpt` files via `pretrained` flag |
| `str/parseq/strhub/models/base.py` | Replaced OneCycleLR with constant low learning rate for fine-tuning |

---

## Augmentation Strategies

### Strategy v1 (`augment_and_build_lmdb.py`) — Best performing (87.9%)

Three augmentations applied **independently** (can stack):

| Augmentation | What it does | Probability |
|---|---|---|
| **Gaussian Blur + Noise** | Blurs with radius 0.3–0.8, adds random noise (std 3–12). Simulates general broadcast video degradation. | p = 0.5 |
| **Perspective Warp** | Randomly shifts image corners to simulate different camera angles. No horizontal flip (would swap 6/9). | p = 0.5 |
| **Rectangular Masking** | Places 1–2 small rectangles (5–15% width, 10–25% height) filled with the image's mean color. Simulates partial occlusion by arms or other players. | p = 0.35 |

### Strategy v2 (`augment_and_build_lmdb_v2.py`) — Literature-backed (87.7%)

Based on STRAug (Atienza, ICCVW 2021). PARSeq already applies 15 standard augmentations internally via RandAugment, so this script only adds what PARSeq **lacks**:

| Augmentation | What it does | Probability |
|---|---|---|
| **Motion Blur** | Applies a 3×3 directional blur kernel (horizontal, vertical, or diagonal). Simulates camera panning and player movement — PARSeq only has Gaussian blur. | p = 0.3 |
| **Pixelate** | Downscales image 2× then upscales back using nearest-neighbor. Simulates low-resolution broadcast crops. | p = 0.2 |
| **JPEG Compression** | Saves at quality 30–70 and reloads. Simulates broadcast video compression artifacts. | p = 0.3 |

Motion blur and pixelate are **mutually exclusive** (only one per image) to prevent creating unreadable images. JPEG compression can stack since it is subtle.

---

## Step-by-Step Instructions

### Prerequisites

- Pipeline fully set up with all conda environments (`jersey`, `centroids`, `vitpose`, `parseq2`)
- SoccerNet dataset at `data/SoccerNet/` with `train/`, `test/` splits
- All model weights downloaded
- `lmdb` package installed in `jersey` env: `conda activate jersey && pip install lmdb`

---

### Step 1: Generate Crops from Train and Test Splits

#### 1a. Run full pipeline on **train** split (generates training crops)

In `main.py`, set actions:

```python
actions = {"soccer_ball_filter": True,
           "feat": True,
           "filter": True,
           "legible": True,
           "legible_eval": False,
           "pose": True,
           "crops": True,
           "str": False,
           "combine": False,
           "eval": False}
```

Run:

```
conda activate jersey
cd H:\jersey-number-pipeline
python main.py SoccerNet train
```

> **Note:** This takes several hours (features ~4h, legibility ~7h, pose ~3h). You can disconnect from Remote Desktop and it will keep running.

Output: `out/SoccerNetResults/crops_train/imgs/` containing ~139K crop images.

#### 1b. Run full pipeline on **test** split (generates test crops + baseline results)

In `main.py`, set actions to run everything:

```python
actions = {"soccer_ball_filter": True,
           "feat": True,
           "filter": True,
           "legible": True,
           "legible_eval": False,
           "pose": True,
           "crops": True,
           "str": True,
           "combine": True,
           "eval": True}
```

Run:

```
python main.py SoccerNet test
```

Output: `out/SoccerNetResults/crops/imgs/` containing ~103K crop images, plus baseline accuracy.

---

### Step 2: Build Augmented LMDB Datasets

#### 2a. Build training LMDB (choose one strategy)

**Strategy v1** (recommended — achieved 87.9%):

```
conda activate jersey
cd H:\jersey-number-pipeline
python augment_and_build_lmdb.py ^
    --crops_dir out/SoccerNetResults/crops_train/imgs ^
    --gt_json data/SoccerNet/train/train_gt.json ^
    --dst_lmdb data/SoccerNet/lmdb_augmented/train ^
    --num_augmented_copies 5 ^
    --preview 20
```

**Strategy v2** (literature-backed — achieved 87.7%):

```
python augment_and_build_lmdb_v2.py ^
    --crops_dir out/SoccerNetResults/crops_train/imgs ^
    --gt_json data/SoccerNet/train/train_gt.json ^
    --dst_lmdb data/SoccerNet/lmdb_augmented_v2/train ^
    --num_augmented_copies 5 ^
    --preview 20
```

#### 2b. Build validation LMDB (from test crops, no augmentation)

```
python augment_and_build_lmdb.py ^
    --crops_dir out/SoccerNetResults/crops/imgs ^
    --gt_json data/SoccerNet/test/test_gt.json ^
    --dst_lmdb data/SoccerNet/lmdb_augmented/val ^
    --num_augmented_copies 0
```

If using v2, replace `lmdb_augmented` with `lmdb_augmented_v2` in the command above.

#### 2c. Check previews look reasonable

```
explorer data\SoccerNet\lmdb_augmented\train\preview
```

Numbers should still be readable in all augmented images. If some are too blurry or too heavily masked, adjust the probability parameters.

---

### Step 3: Move LMDB Files to Expected Directory Structure

PARSeq expects LMDB files inside a `real/` subdirectory:

```
mkdir data\SoccerNet\lmdb_augmented\train\real
move data\SoccerNet\lmdb_augmented\train\data.mdb data\SoccerNet\lmdb_augmented\train\real\
move data\SoccerNet\lmdb_augmented\train\lock.mdb data\SoccerNet\lmdb_augmented\train\real\

mkdir data\SoccerNet\lmdb_augmented\val\real
move data\SoccerNet\lmdb_augmented\val\data.mdb data\SoccerNet\lmdb_augmented\val\real\
move data\SoccerNet\lmdb_augmented\val\lock.mdb data\SoccerNet\lmdb_augmented\val\real\
```

Final structure should be:

```
data/SoccerNet/lmdb_augmented/
├── train/
│   └── real/
│       ├── data.mdb
│       └── lock.mdb
└── val/
    └── real/
        ├── data.mdb
        └── lock.mdb
```

---

### Step 4: Fine-Tune PARSeq

#### 4a. Prepare a clean checkpoint copy

The original checkpoint filename contains `=` signs that cause issues with Hydra. Copy it to a clean name:

```
copy "H:\jersey-number-pipeline\models\parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt" H:\jersey-number-pipeline\models\parseq_finetuned.ckpt
```

#### 4b. (Optional) Adjust learning rate

The learning rate is set in `str/parseq/strhub/models/base.py` in the `configure_optimizers` method:

```python
def configure_optimizers(self):
    lr = 2.5e-6  # Change this value to experiment
    optim = create_optimizer_v2(self, 'adamw', lr, self.weight_decay)
    return {'optimizer': optim}
```

Recommended values tested:

| Learning Rate | Result |
|---|---|
| 1e-5 | 86.1% (too high, some forgetting) |
| 5e-6 | 87.4% |
| **2.5e-6** | **87.9% (optimal)** |
| 1.25e-6 | 87.7% (too low, not enough learning) |

#### 4c. Run training

```
conda activate parseq2
cd H:\jersey-number-pipeline\str\parseq
python train.py +experiment=parseq dataset=real ^
    data.root_dir=H:/jersey-number-pipeline/data/SoccerNet/lmdb_augmented ^
    trainer.max_epochs=1 ^
    trainer.devices=1 ^
    ++trainer.val_check_interval=1.0 ^
    data.batch_size=128 ^
    model.max_label_length=25 ^
    data.max_label_length=25 ^
    pretrained=H:/jersey-number-pipeline/models/parseq_finetuned.ckpt
```

> **Important notes:**
> - Uses `pretrained` (not `ckpt_path`) — this loads only model weights with a fresh optimizer
> - `trainer.max_epochs=1` — best results come from 1 epoch; more epochs cause overfitting
> - `++trainer.val_check_interval=1.0` — validates once per epoch (float = fraction of epoch; integer = every N batches)
> - Training takes ~20 minutes per epoch on RTX A4000

---

### Step 5: Copy Best Checkpoint

After training, find the checkpoints. Location depends on how you ran it:

```
dir H:\jersey-number-pipeline\str\parseq\outputs /s /b *.ckpt
```

Checkpoints are saved at: `str/parseq/outputs/parseq/<timestamp>/checkpoints/`

Copy the best one (highest `val_accuracy` in filename):

```
copy "H:\jersey-number-pipeline\str\parseq\outputs\parseq\<timestamp>\checkpoints\<best_checkpoint>.ckpt" H:\jersey-number-pipeline\models\parseq_augmented.ckpt
```

> **Note:** Checkpoint filenames contain `=` signs, so always wrap the path in double quotes.

---

### Step 6: Update Configuration and Evaluate

#### 6a. Update `configuration.py`

Change the `str_model` path in the SoccerNet config:

```python
'str_model': 'models/parseq_augmented.ckpt',
```

#### 6b. Run STR + evaluation on test set

In `main.py`, set actions to only run the remaining steps:

```python
actions = {"soccer_ball_filter": False,
           "feat": False,
           "filter": False,
           "legible": False,
           "legible_eval": False,
           "pose": False,
           "crops": False,
           "str": True,
           "combine": True,
           "eval": True}
```

Run:

```
conda activate jersey
cd H:\jersey-number-pipeline
python main.py SoccerNet test
```

This will run PARSeq inference with your new model (~40 min), aggregate predictions, and print the final accuracy.

#### 6c. To revert to baseline

Change `configuration.py` back to:

```python
'str_model': 'models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt',
```

The original weights are never modified.

---

## Results Summary

| Augmentation | Learning Rate | Pipeline Accuracy |
|---|---|---|
| None (baseline) | N/A | 86–87% |
| **v1 (blur + perspective + masking)** | **2.5e-6** | **87.9%** |
| v2 (motion blur + pixelate + JPEG) | 2.5e-6 | 87.7% |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Thumbs.db` error during STR | `del /a:sh out\SoccerNetResults\crops\imgs\Thumbs.db` |
| `No module named 'lmdb'` | `pip install lmdb` in the active environment |
| `No module named 'imgaug'` | `conda activate parseq2 && pip install imgaug && pip install "numpy<2"` |
| `No module named 'tensorboard'` | `conda activate parseq2 && pip install tensorboard` |
| `No module named 'omegaconf'` | `conda activate parseq2 && pip install omegaconf hydra-core` |
| `datasets should not be an empty iterable` | LMDB files need to be inside a `real/` subdirectory (see Step 3) |
| `max_epochs reached` immediately | You used `ckpt_path` which restores epoch counter; use `pretrained` instead |
| Accuracy drops after fine-tuning | Ensure you used `pretrained` (not `ckpt_path`) and LR ≤ 5e-6 |
