## Real-ESRGAN Enhancement Strategy

### Overview

Over 94% of SoccerNet player crops have a height below 64px. At this resolution, jersey numbers are often too blurry for reliable recognition.

We explored using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — a blind super-resolution model that upscales images 4x while recovering sharp edges and fine details — to improve recognition accuracy.

**Attempt 1 — Apply ESRGAN at inference only:**
Apply Real-ESRGAN to crops at inference time, without retraining PARSeq. Result: accuracy **dropped to ~45%**. The pre-trained PARSeq model was fine-tuned on raw low-resolution crops, so feeding it ESRGAN-enhanced high-resolution images caused a training/inference distribution mismatch. See [report_real_esrgan.md](report_real_esrgan.md) for a detailed analysis.

**Attempt 2 — Apply ESRGAN consistently at both training and inference:**
Re-process the entire training LMDB with Real-ESRGAN, re-fine-tune PARSeq on the enhanced images, then apply ESRGAN at inference (STR). Result: accuracy **recovered to ~85%**, with the distribution mismatch eliminated.

### What Changed

Compared to the original codebase, the following files were added or modified:

**New files:**

| File | Description |
|---|---|
| `esrgan_lmdb.py` | Applies Real-ESRGAN to every image in the training LMDB and writes a new LMDB |
| `real_esrgan_upsampler.py` | Wraps `RealESRGANer` for use in the inference pipeline |
| `plot_crop_analysis.py` | Plots the height distribution of crop images |
| `report_real_esrgan.md` | Analysis report on the Real-ESRGAN enhancement experiments |

**Crop size analysis** (used to justify the ESRGAN strategy):

| File | Description |
|---|---|
| `crop size analysis/check_crop_sizes.py` | Script to compute crop height statistics |
| `crop size analysis/crop_height_distribution.png` | Histogram of crop heights in the test set |
| `crop size analysis/crop_size_analysis.txt` | Full crop height distribution statistics |

**Modified files:**

| File | Description |
|---|---|
| `str.py` | Added `--use_esrgan` flag to apply ESRGAN to all crops before PARSeq recognition |
| `str/parseq/train.py` | Added support for loading local `.ckpt` checkpoints as pretrained weights; truncates `pos_queries` to match current `max_label_length` |
| `main.py` | STR inference command controlled by `upsampling` flag; training uses `lmdb_esrgan` |
| `helpers.py` | Added `generate_crops_with_upsampling()` with optional ESRGAN; added pose fallback for unreliable keypoints |
| `configuration.py` | Added `numbers_data_esrgan` path; updated `str_model` to point to ESRGAN-trained checkpoint |
| `setup.py` | Minor updates for compatibility |

## Setup

For Linux/Mac:
```
cd jersey-number-pipeline
source SetupEnv.sh
```
For Windows:
```
cd jersey-number-pipeline
SetupEnv.bat
```
These scripts will download dataset, install dependencies and configure the repository.

## Download Pre-trained Models and Weights

### 1. PARSeq fine-tuned checkpoint

Download the fine-tuned PARSeq model from Google Drive and place it under `models/`:

https://drive.google.com/file/d/1T7dzz32KywFApvuzDCrnmsa2_62Lxkr0/view?usp=drive_link

### 2. Real-ESRGAN weights

Download the official Real-ESRGAN x4 weights directly from the original repository:

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights/
```

### 3. Verify

After both downloads, confirm the following files exist:
```
models/parseq_soccer_esrgan.ckpt
weights/RealESRGAN_x4plus.pth
```

## Inference

Run inference on the test set:
```
python3 main.py SoccerNet test
```

---

## Training (SoccerNet with Real-ESRGAN) (optional)

This section describes how to reproduce the fine-tuned PARSeq model from scratch.

### Step 1: Prepare Real-ESRGAN-processed training data

Install dependencies in a separate environment and process the training LMDB:

```bash
conda create -n esrgan python=3.9 -y
conda activate esrgan
pip install "numpy<2" realesrgan basicsr "torchvision<0.16" lmdb pillow tqdm
python esrgan_lmdb.py --src_root data/SoccerNet/lmdb --dst_root data/SoccerNet/lmdb_esrgan --model_path weights/RealESRGAN_x4plus.pth
```

This applies Real-ESRGAN 4x upscaling to every image in the training LMDB and saves the results to `data/SoccerNet/lmdb_esrgan/`. This step can take several hours.

### Step 2: Fine-tune PARSeq on ESRGAN-processed data

```bash
conda activate SoccerNet
pip install pandas
python3 main.py SoccerNet train --train_str
```

Checkpoints are saved under `str/parseq/outputs/`. Select the best checkpoint by `val_accuracy` and copy it to `models/`:

```bash
cp str/parseq/outputs/<run_dir>/checkpoints/<best>.ckpt models/parseq_soccer_esrgan.ckpt
```

### Step 3: Update configuration

In `configuration.py`, set:
```python
'str_model': 'models/parseq_soccer_esrgan.ckpt',
```

### Step 4: Run inference

```bash
python3 main.py SoccerNet test
```

Real-ESRGAN is applied to all crops at inference time before PARSeq recognition, matching the training distribution.


## Troubleshooting

### `RuntimeError: Numpy is not available` or NumPy 2.x conflicts

Several packages in this project require NumPy < 2. If you see this error, downgrade NumPy first before reinstalling torch:

```bash
conda activate SoccerNet
pip install "numpy<2"
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

For the `esrgan` environment:
```bash
conda activate esrgan
pip install "numpy<2" "torchvision<0.16"
```

---

### `ModuleNotFoundError: No module named 'basicsr'` during STR inference

`basicsr` and `realesrgan` must be installed in the `parseq2` conda environment (used for STR inference), not just the `esrgan` environment:

```bash
conda run -n parseq2 pip install "numpy<2" realesrgan basicsr "torchvision<0.16"
```

---

### `ModuleNotFoundError: No module named 'pkg_resources'`

Install or downgrade setuptools:

```bash
pip install setuptools==59.5.0
```

---

### Hydra error: `ValueError: Error parsing override` with checkpoint path containing `=`

PARSeq checkpoint filenames often contain `=` (e.g. `epoch=17-step=1854-val_accuracy=95.8104.ckpt`), which Hydra interprets as a key-value separator. Copy the checkpoint to a name without `=`:

```bash
cp str/parseq/outputs/<run_dir>/checkpoints/<best>.ckpt models/parseq_soccer_esrgan.ckpt
```

Then set `str_model` in `configuration.py` to point to the new path.

---

### `torchvision.transforms.functional_tensor` not found

This error appears when `torchvision >= 0.16` is installed in the `esrgan` environment. Downgrade:

```bash
conda activate esrgan
pip install "torchvision<0.16"
```

---

### ViTPose killed / out of memory

ViTPose processes all images in the dataset and is memory-intensive. If the process is killed:
- Ensure no other large jobs are running on the same GPU
- The pose step (105k+ images) takes approximately 2–3 hours on a single GPU — this is expected

---

### `pandas` not found when running training

```bash
conda run -n SoccerNet pip install pandas
```

---

### LMDB data not downloaded

The weakly-labelled jersey number crops LMDB can be downloaded with `gdown`:

```bash
pip install gdown
gdown <file_id>  # see data download links in the Data section below
```

Extract with Python if `unzip` is unavailable:
```python
import zipfile
with zipfile.ZipFile('lmdb.zip', 'r') as z:
    z.extractall('data/SoccerNet/')
```