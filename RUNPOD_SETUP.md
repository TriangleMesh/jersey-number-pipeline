# RunPod Ishaan notes

## Baseline result
- SoccerNet test tracklets: 1211
- Correct: 1053
- Accuracy: 86.95293146160198%

## Important fixes
- Replaced conda-run based stage launches in `main.py` with direct env Python paths:
  - `/workspace/miniconda3/envs/centroids/bin/python`
  - `/workspace/miniconda3/envs/vitpose/bin/python`
  - `/workspace/miniconda3/envs/parseq2/bin/python`
- Added compatibility patch for older checkpoints in:
  - `centroid_reid.py`
  - `str.py`
- Adjusted setup flow for RunPod and `/workspace`-based environments.

## Important environment note
Install conda in `/workspace/miniconda3`, not `/root`, to avoid container disk issues.
