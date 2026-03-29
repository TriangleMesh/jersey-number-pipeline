Experiment: PARSeq fine-tuning on crop-based SoccerNet train LMDB with sports-specific augmentation

Setup:
- Training data built from train split crops, not raw frames
- Validation split created from train crops by tracklet
- Augmentations included motion blur, pixelation/compression-style degradation, and related sports-specific perturbations

Results:
- Baseline public test accuracy: 86.95% (1053 / 1211)
- Raw-frame fine-tune accuracy: 33.94% (411 / 1211)
- Crop-based fine-tune accuracy: 84.72% (1026 / 1211)

Interpretation:
- Crop-based training is far more sensible than raw-frame training.
- However, this fine-tuning setup still did not outperform the provided baseline checkpoint.
- Likely reasons include:
  1. baseline checkpoint may already be strongly optimized for SoccerNet-style crops
  2. crop labels are still noisy at frame level even when tracklet labels are correct
  3. added augmentation may be too strong or not aligned enough with the best baseline recipe

Conclusion:
- Crop-based fine-tuning is viable but did not beat baseline in current form.
- Baseline checkpoint remains the best-performing model for the assignment.
