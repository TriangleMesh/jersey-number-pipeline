
Tiny Crop Upscaling & Pipeline Stability Guide


This branch focuses on stabilizing the SoccerNet jersey number pipeline and introducing a lightweight OCR-side improvement.

Baseline pipeline:

ReID → Legibility → Pose → Crop → PARSeq → Voting

The goal was:
	•	reliably reproduce baseline results
	•	introduce a small, controlled improvement at the OCR stage

Final accepted change:
	•	Gated tiny crop upscaling before PARSeq

⸻

What Changed

File	Change
str.py	Added gated tiny upscaling before OCR inference
main.py	Replaced unstable conda run calls with fixed interpreter paths
centroid_reid.py, str.py	Added PyTorch load compatibility patch
legibility_classifier.py	Fixed SAM import path
.gitignore	Cleaned output + checkpoint artifacts


⸻

Key Improvement: Gated Tiny Upscaling

Implementation (str.py)

def maybe_upscale_tiny_crop(image, min_h=64, scale_factor=2):
    w, h = image.size
    if h >= min_h:
        return image

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
    return image

Applied before OCR:

image = maybe_upscale_tiny_crop(image, min_h=64, scale_factor=2)


⸻

Motivation
	•	PARSeq struggles with very small crops (<64 px)
	•	Digits become compressed → strokes indistinguishable
	•	Simple upscaling increases effective resolution without heavy models

⸻

Result

Setting	Accuracy
Baseline	86.95%
+ Tiny Upscaling	87.20%

Improvement: +0.25

⸻

Critical Reproduction Notes (IMPORTANT)

1. Clear Cached Outputs

Pipeline skips stages if outputs exist

Before running:

rm -rf out/SoccerNetResults/

Otherwise:
	•	results may not reflect code changes
	•	pipeline appears correct but is using stale outputs

⸻

2. Use Correct Environments

Separate environments were used:
	•	jersey → main pipeline
	•	parseq2 → OCR

Common failure:
	•	running scripts in wrong env → import errors

⸻

3. Avoid conda run

Replaced:

conda run -n parseq2 python str.py ...

With:

/workspace/miniconda3/envs/parseq2/bin/python str.py ...

Reason:
	•	conda run was unreliable in RunPod
	•	sometimes used wrong environment silently

⸻

4. PyTorch Compatibility Patch

_orig_torch_load = torch.load

def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_compat

Fixes checkpoint loading issues in newer PyTorch versions.

⸻

Major Issues Encountered

Environment
	•	No CUDA on local machine → could not run full pipeline locally
	•	RunPod environments inconsistent across sessions
	•	Multiple conda envs caused execution mistakes

⸻

Dependencies

Frequent issues with:
	•	torchvision mismatches
	•	cv2 missing
	•	pandas missing
	•	timm warnings
	•	SAM import path mismatch

⸻

Pipeline
	•	Dataset structure had to match expected layout exactly
	•	Manual data setup prone to errors
	•	Cached outputs caused misleading results

⸻

Critical Bug: Confidence Format
	•	Some stages returned float
	•	Others expected list

This broke:
	•	helpers.py aggregation
	•	fallback logic

Fix:
	•	standardized confidence format before post-processing

⸻

Removed / Abandoned Work

Real-ESRGAN

Attempted for super-resolution.

Rejected due to:
	•	dependency conflicts (basicsr, realesrgan)
	•	slow inference
	•	unstable integration

⸻

Fallback Scan

Attempted for low-confidence predictions.

Rejected due to:
	•	inconsistent confidence handling
	•	pipeline instability
	•	complex integration


Key Takeaways
	•	Most effort went into engineering/debugging, not modeling
	•	Simple preprocessing > complex unstable modules
	•	Cache handling is critical in multi-stage pipelines
	•	Data format consistency is essential
	•	Environment reproducibility is a major challenge

Final Result
	•	Baseline reproduced successfully
	•	One stable improvement added:

Gated tiny upscaling (<64 px crops)

Final accuracy:
	•	86.95% → 87.20%
