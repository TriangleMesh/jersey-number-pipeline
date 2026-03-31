
# Jersey Number Pipeline — Raunak Run Plan (SoccerNet)

## Goal 
I’m trying to run the full **jersey-number pipeline** end-to-end on the **SoccerNet dataset** (train/test/challenge), so that:

1. The pipeline completes without crashes (models + data paths + env all correct)
2. It outputs **jersey ID predictions** for tracklets (per-player/tracklet)
3. I can compare accuracy / quality of the outputs vs the original baseline
4. I can package the changes into a clean PR (tracklet consolidation + robustness)

---

## High-level pipeline (what the code does)
For a given SoccerNet split (`train`, `test`, `challenge`):

1. **Load tracklets/images** from `data/SoccerNet/<split>/images`
2. **Detect/ignore soccer balls** (filter step)
3. **Generate ReID features** for tracklets (centroids-reid model)
4. **Run STR/OCR** on jersey crops to get digit predictions + confidences
5. **Combine / consolidate predictions per tracklet**
6. Write results to an output JSON (jersey id result file)

---

## What I changed (my contribution / direction)
### 1) Combine step uses my tracklet consolidation
Instead of directly using the old `helpers.process_jersey_id_predictions(...)`,
the combine step calls my improved function:
- `tracklet_consolidation.process_jersey_id_predictions_v2(...)`

### 2) Better consolidation strategy (intent)
The goal is to reduce noisy OCR/STR outputs by:
- normalizing common OCR mistakes (like O→0, I→1, etc.)
- using confidence-weighted voting / aggregation
- allowing abstain (-1) when evidence is weak (optional)
- fallback to legacy helper logic if v2 parsing fails (optional)

> NOTE: This is **not guaranteed** to improve accuracy in every case.
> It’s meant to be more robust and should be validated by running + comparing outputs.

---

## Environment: how I’m running it
### Why GPU?
This pipeline is heavy (models + lots of images). My Mac is not enough.
So I run on a TensorDock GPU VM.

### Recommended setup (TensorDock)
- GPU: **V100 32GB** is usually enough (A100 is faster but more expensive)
- OS: **Ubuntu 22.04 LTS** (NVIDIA driver image preferred)
- Use `tmux` so runs survive SSH disconnects.

---

## One-time setup on the VM (important)
### 1) SSH in
From Mac:
```bash
ssh -i ~/.ssh/tensordock_ed25519 user@<VM_IP>
````

### 2) Use tmux (so the run doesn’t die if VS Code/SSH disconnects)

```bash
tmux new -s jersey
# detach: Ctrl+b then d
# later reattach:
tmux attach -t jersey
```

### 3) Go to repo

```bash
cd ~/jersey-number-pipeline
git status
```

### 4) Activate conda env

```bash
conda activate SoccerNet
```

### 5) Confirm GPU is visible

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset layout (what the code expects vs what we actually have)

The code expects:

* `data/SoccerNet/train/images`
* `data/SoccerNet/test/images`
* `data/SoccerNet/val/images`

But sometimes the dataset has:

* `train/images`, `test/images`, `challenge/images`
  and **no `val/`**.

### Fix: map `val/images` → `challenge/images`

```bash
rm -rf data/SoccerNet/val
mkdir -p data/SoccerNet/val
ln -s ../challenge/images data/SoccerNet/val/images
ls data/SoccerNet/val/images | head
```

---

## Model files (common failure)

A frequent failure is missing ReID checkpoint, e.g.:
`market1501_resnet50_256_128_epoch_120.ckpt`

### Check it exists:

```bash
ls -lh reid/centroids-reid/models/market1501_resnet50_256_128_epoch_120.ckpt
```

If missing, download it (repo scripts may do this, or download manually if needed).

---

## Running the pipeline (my standard run commands)

### Run SoccerNet test split (log everything)

```bash
python main.py SoccerNet test 2>&1 | tee test_run.log
```

### Watch progress

```bash
tail -f test_run.log
```

### Run train split (bigger / slower)

```bash
python main.py SoccerNet train 2>&1 | tee train_run.log
```

---

## Outputs (what I expect to get)

After a successful run, I expect:

* Output folders like `out/SoccerNetResults/<split>/...`
* A jersey id result JSON file (location depends on configuration)
* Logs showing:

  * “determine soccer ball”
  * “generate features”
  * “classifying legibility”
  * “combine tracklet results”
  * “writing jersey_id_result”

---

## Common issues + quick fixes

### 1) “Broken pipe” / “connection closed”

This is usually **SSH/VS Code disconnect**, not pipeline failure.
Fix: run inside `tmux`.

### 2) Missing `val/images`

Fix: symlink val → challenge (see above).

### 3) Missing model `.ckpt`

Fix: download the checkpoint into:
`reid/centroids-reid/models/`

### 4) KeyError like `KeyError: 'jersey_id_result'`

This means `configuration.py` is missing a config key for the split.
Fix: add a `jersey_id_result` entry for the split you’re running.

### 5) Conda channel “Terms of Service not accepted”

Fix example:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## Accuracy plan (how I’ll prove it’s better)

I can’t “guarantee” best accuracy without measurement.
So the plan is:

1. Run baseline method (old helper consolidation) on a split
2. Run v2 consolidation on the same split
3. Compare:

   * % correct jersey IDs (if ground truth available)
   * of abstains (-1) vs wrong confident guesses
   * consistency across frames/tracklets
4. Keep v2 only if it improves metrics or reduces obvious errors

---

## Collaboration plan (when to ask a friend to test)

Ask a friend to test **after**:

* pipeline runs end-to-end on my VM at least once
* model + dataset paths are stable
* output JSON is produced correctly

What I want them to test:

* fresh clone → setup → run on their machine/VM
* confirm they can reproduce the same run and outputs
* confirm no missing-file surprises

---

## Current status (checkpoint)

- Repo runs on TensorDock VM
- Dataset paths fixed (val → challenge mapping)
- eID checkpoint downloaded + found
- pipe line starts + progresses through feature generation
- Need stable long-running session via tmux + confirm final output JSON
- Need clean accuracy comparison vs baseline consolidation

---

## Next steps (what I do next)

1. Run `SoccerNet test` fully to completion (in tmux)
2. Confirm output JSON and log contains “combine step finished”
3. Run baseline vs v2 consolidation comparison
4. Write a short PR description explaining:

   * why consolidation changed
   * how it was tested
   * results/observations


