Experiment: augmented PARSeq fine-tuning on LMDB built from raw SoccerNet train tracklet frames

Result:
- Baseline test accuracy: 86.95% (1053 / 1211)
- Augmented PARSeq test accuracy: 33.94% (411 / 1211)

Likely cause:
- STR inference uses cropped jersey/torso patches.
- Fine-tuning data was built from raw train tracklet frames, not the final cropped patches.
- This created a train/test representation mismatch and likely heavy label noise.

Conclusion:
- The experiment does not show that augmentation is bad.
- It shows that fine-tuning STR on raw train frames is the wrong target format.
