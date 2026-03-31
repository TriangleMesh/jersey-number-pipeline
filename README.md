# A General Framework for Jersey Number Recognition in Sports

Code, data, and model weights for paper [A General Framework for Jersey Number Recognition in Sports](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf) (Maria Koshkina, James H. Elder).

![Pipeline](docs/soccer_pipeline.png)

The pipeline performs tracklet-level jersey number recognition on the SoccerNet dataset through the following stages:

1. **Soccer ball filtering** — removes ball tracklets by image size
2. **ReID feature extraction** — generates appearance features using Centroids-ReID (ResNet50)
3. **Outlier removal** — fits a Gaussian to ReID features and removes frames >3.5σ from the mean
4. **Legibility classification** — ResNet34 binary classifier filters out illegible frames
5. **Pose estimation** — ViTPose-Huge detects body keypoints to locate the torso region
6. **Torso cropping** — extracts jersey region using shoulder and hip keypoints
7. **Scene text recognition** — PARSeq reads the jersey number from each crop
8. **Tracklet aggregation** — confidence-weighted majority voting produces the final prediction

---

## Setup

**Prerequisites:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda must be installed.

### Quick Setup

The setup scripts download the SoccerNet dataset, clone dependency repos, create conda environments, install all packages, and download model weights.

**Linux/Mac:**
```bash
cd jersey-number-pipeline
source SetupEnv.sh
```

**Windows:**
```
cd jersey-number-pipeline
SetupEnv.bat
```

### What the Setup Scripts Do

| Step | Description |
|---|---|
| `SetupSoccerNetDataset.py` | Downloads the SoccerNet Jersey 2023 dataset via the SoccerNet API, renames and unzips it to `data/SoccerNet/` |
| `setup.py SoccerNet` | Clones sub-repos (SAM, Centroids-ReID, ViTPose, PARSeq), creates separate conda environments for each, installs dependencies, and downloads model weights |

The pipeline uses **four conda environments** because each sub-model has different dependency requirements:

| Environment | Python | Purpose |
|---|---|---|
| `jersey` / `SoccerNet` | 3.9–3.10 | Main pipeline, legibility classifier |
| `centroids` | 3.8–3.9 | Centroids-ReID feature extraction |
| `vitpose` | 3.8 | ViTPose pose estimation |
| `parseq2` | 3.9 | PARSeq scene text recognition |

### Manual Setup

If the automated setup fails, you can set up each component manually. Refer to `setup.py` for the exact packages and versions needed for each environment.

**Sub-repositories:**

| Component | Location | Source |
|---|---|---|
| SAM | `jersey-number-pipeline/sam` | [https://github.com/davda54/sam](https://github.com/davda54/sam) |
| Centroids-ReID | `jersey-number-pipeline/reid/centroids-reid` | [https://github.com/mikwieczorek/centroids-reid](https://github.com/mikwieczorek/centroids-reid) |
| ViTPose | `jersey-number-pipeline/pose/ViTPose` | [https://github.com/ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose) |
| PARSeq | `jersey-number-pipeline/str/parseq` | [https://github.com/baudm/parseq](https://github.com/baudm/parseq) |

**Model weights** (download and place in the listed directories):

| Model | Download | Location |
|---|---|---|
| Centroids-ReID | [market_resnet50](https://drive.google.com/file/d/1bSUNpvMfJkvCFOu-TK-o7iGY1p-9BxmO/view?usp=sharing) | `reid/centroids-reid/models/market1501_resnet50_256_128_epoch_120.ckpt` |
| ViTPose-Huge | [vitpose-h](https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe) | `pose/ViTPose/checkpoints/vitpose-h.pth` |
| PARSeq (SoccerNet) | [parseq_finetuned](https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view?usp=sharing) | `models/` |
| Legibility (SoccerNet) | [resnet34](https://drive.google.com/file/d/18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw/view?usp=sharing) | `models/` |

> **Note:** The ReID weights downloaded via `gdown` may sometimes download an HTML page instead of the actual checkpoint. If the file is only a few KB instead of ~307MB, delete it and re-download manually from the Google Drive link.

---

## Running the Pipeline

### Inference on Test Set

```
python main.py SoccerNet test
```

This runs the full pipeline and prints the final accuracy against the test set ground truth.

### Selective Step Execution

Edit the `actions` dictionary in `main.py` to skip completed steps. For example, to re-run only STR and evaluation:

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

### Configuration

Update `configuration.py` to set custom paths to data, models, or dependencies.

---

## Common Issues & Fixes

### CUDA Compatibility

**Error:** `NVIDIA RTX A4000 with CUDA capability sm_86 is not compatible with the current PyTorch installation`

The default setup installs PyTorch builds for older CUDA compute capabilities. Fix by upgrading PyTorch in the affected environment:

```
conda activate <env_name>
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

This applies to `centroids`, `vitpose`, and `parseq2` environments if running on newer GPUs (RTX 3000/4000 series, A4000, etc.).

### NumPy Version Conflicts

**Error:** `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0`

```
pip install "numpy<2"
```

### PyTorch Lightning Version Mismatch

**Error:** `ImportError: cannot import name 'EPOCH_OUTPUT' from 'pytorch_lightning.utilities.types'`

```
conda activate parseq2
pip install "pytorch-lightning==1.9.5"
```

### Missing Packages

| Error | Fix |
|---|---|
| `No module named 'tqdm'` | `pip install tqdm` in the affected environment |
| `No module named 'imgaug'` | `pip install imgaug` then `pip install "numpy<2"` |
| `No module named 'tensorboard'` | `pip install tensorboard` |
| `No module named 'omegaconf'` | `pip install omegaconf hydra-core` |

### Corrupted Model Downloads

**Error:** `_pickle.UnpicklingError: invalid load key, '<'.`

The downloaded file is an HTML page from Google Drive, not the actual weights. Delete the file and re-download manually from the Google Drive link in your browser, or use `gdown` with the file ID:

```
gdown 1bSUNpvMfJkvCFOu-TK-o7iGY1p-9BxmO -O reid/centroids-reid/models/market_resnet50.pth
```

### Windows-Specific

| Issue | Fix |
|---|---|
| `Thumbs.db` errors during STR inference | `del /a:sh out\SoccerNetResults\crops\imgs\Thumbs.db` |
| Path separator issues | The pipeline uses mixed `/` and `\` separators; this is handled automatically |

---

## Individual Contributions

Each team member's work is on a separate branch with its own README containing detailed setup and reproduction instructions.

| Branch | Author | Description | Test Accuracy |
|---|---|---|---|
| `main` | — | Original baseline pipeline | 86–87% |
| `data-augmentation-aarav` | [Aarav Gosalia] | Data augmentation + PARSeq fine-tuning optimization | **87.9%** |
| `real-esrgan-qingyun` | [Qingyin Qian] | Real-ESRGAN super-resolution preprocessing | Attempt 1: 45%; Attempt 2: 85.5% |
| `runpod-ishaan` | [Ishaan Singh Yadav] | Bicubic gated resize preprocessing | 87.2% |
| `raunak-khanna` | [Raunak Khanna] | - | - |

> **To review a branch:** `git checkout <branch-name>` and refer to that branch's `README.md` for full instructions and working demo.

---

## Training

### Train Legibility Classifier (Hockey)
```
python legibility_classifier.py --train --arch resnet34 --sam --data <dataset-directory> --trained_model_path ./experiments/hockey_legibility.pth
```

### Fine-tune PARSeq STR (Hockey)
```
python main.py Hockey train --train_str
```

### Fine-tune Legibility Classifier (SoccerNet)

Weakly-labelled datasets are generated first using models trained on Hockey data:

```
python legibility_classifier.py --finetune --arch resnet34 --sam --data <dataset-directory> --full_val_dir <dataset-directory>/val --trained_model_path ./experiments/hockey_legibility.pth --new_trained_model_path ./experiments/sn_legibility.pth
```

### Fine-tune PARSeq STR (SoccerNet)
```
python main.py SoccerNet train --train_str
```

Trained models are saved under `str/parseq/outputs`.

---

## Data

**SoccerNet Jersey Number Recognition:** [https://github.com/SoccerNet/sn-jersey](https://github.com/SoccerNet/sn-jersey)

Download and save under the `data/` subfolder, or use `SetupSoccerNetDataset.py`.

Additional datasets:
- [Weakly-labelled player images](https://drive.google.com/file/d/1CmJfUmS_ZudgEiCT14b2CbyMA3nEO_uy/view?usp=sharing) (legibility classifier training)
- [Weakly-labelled jersey number crops in LMDB](https://drive.google.com/file/d/1PX8XDF3nNMZAvcjL6M5hurwX78ePAhSs/view?usp=sharing) (STR fine-tuning)
- Hockey dataset: contact [Maria Koshkina](mailto:koshkina@hotmail.com?subject=Hockey)

---

## Citation

```
@InProceedings{Koshkina_2024_CVPR,
    author    = {Koshkina, Maria and Elder, James H.},
    title     = {A General Framework for Jersey Number Recognition in Sports Video},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3235-3244}
}
```

## Acknowledgements

- [PARSeq](https://github.com/baudm/parseq)
- [Centroid-Reid](https://github.com/mikwieczorek/centroids-reid)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [SoccerNet](https://github.com/SoccerNet/sn-jersey)
- [McGill Hockey Player Tracking Dataset](https://github.com/grant81/hockeyTrackingDataset)
- [SAM](https://github.com/davda54/sam)

## License

[![License](https://i.creativecommons.org/l/by-nc/3.0/88x31.png)](http://creativecommons.org/licenses/by-nc/3.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 3.0 Unported License](http://creativecommons.org/licenses/by-nc/3.0/).
