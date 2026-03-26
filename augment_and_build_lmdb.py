"""
augment_and_build_lmdb.py
─────────────────────────
Reads an existing PARSeq LMDB training set, applies data augmentations,
and writes a new, enlarged LMDB that can be used to fine-tune PARSeq.

Augmentations (each applied independently with configurable probability):
  1. Gaussian blur + additive Gaussian noise
  2. Perspective / affine warps
  3. Random rectangular masking  ← simulates digit occlusion (arm, body)

Usage
─────
  # From the jersey-number-pipeline root:
  python augment_and_build_lmdb.py ^
      --src_lmdb  data/SoccerNet/lmdb/train ^
      --dst_lmdb  data/SoccerNet/lmdb_augmented/train ^
      --num_augmented_copies 5

  # Or, build LMDB from crop images + ground truth JSON:
  python augment_and_build_lmdb.py ^
      --crops_dir  out/SoccerNetResults/crops_train/imgs ^
      --gt_json    data/SoccerNet/train/train_gt.json ^
      --dst_lmdb   data/SoccerNet/lmdb_augmented/train ^
      --num_augmented_copies 5

After building, fine-tune PARSeq:
  conda run --live-stream -n parseq2 python str/parseq/train.py ^
      +experiment=parseq dataset=real ^
      data.root_dir=data/SoccerNet/lmdb_augmented ^
      trainer.max_epochs=25 pretrained=parseq ^
      trainer.devices=1 data.batch_size=128 data.max_label_length=2
"""

import argparse
import io
import os
import random
import sys
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

# ──────────────────────────────────────────────────────────────────────
#  Augmentation functions
# ──────────────────────────────────────────────────────────────────────

def apply_gaussian_blur_and_noise(img, blur_radius_range=(0.3, 0.8), noise_std_range=(3, 12)):
    """Gaussian blur + additive Gaussian noise."""
    # Blur
    radius = random.uniform(*blur_radius_range)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Noise
    arr = np.array(img, dtype=np.float32)
    std = random.uniform(*noise_std_range)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_perspective_transform(img, distortion_scale=0.15):
    """Random perspective warp (no horizontal flip — would swap 6/9)."""
    w, h = img.size
    half_w, half_h = w / 2, h / 2

    # Four corners with random offsets
    def jitter(x, y):
        dx = random.uniform(-distortion_scale * half_w, distortion_scale * half_w)
        dy = random.uniform(-distortion_scale * half_h, distortion_scale * half_h)
        return (x + dx, y + dy)

    # Source corners (original)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    # Destination corners (warped)
    dst = [jitter(*p) for p in src]

    # Compute perspective coefficients
    coeffs = _find_perspective_coeffs(dst, src)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def _find_perspective_coeffs(src_pts, dst_pts):
    """Compute 8 perspective transform coefficients."""
    import numpy as np
    matrix = []
    for (x, y), (X, Y) in zip(src_pts, dst_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([p for pt in dst_pts for p in pt], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.flatten())


def apply_random_rectangular_mask(img, num_masks_range=(1, 2),
                                   mask_width_ratio=(0.05, 0.15),
                                   mask_height_ratio=(0.10, 0.25)):
    """
    Random rectangular masking — simulates one digit being occluded
    by an arm, body part, or another player.

    Fills rectangular patches with the image's mean color (less jarring
    than pure black, avoids teaching the model to key on black patches).
    """
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Use mean color so the mask blends more naturally
    mean_color = tuple(arr.mean(axis=(0, 1)).astype(np.uint8).tolist())
    draw = ImageDraw.Draw(img)

    num_masks = random.randint(*num_masks_range)
    for _ in range(num_masks):
        mw = int(w * random.uniform(*mask_width_ratio))
        mh = int(h * random.uniform(*mask_height_ratio))
        x0 = random.randint(0, max(w - mw, 0))
        y0 = random.randint(0, max(h - mh, 0))
        draw.rectangle([x0, y0, x0 + mw, y0 + mh], fill=mean_color)

    return img


def augment_image(img, p_blur=0.5, p_perspective=0.5, p_mask=0.35):
    """Apply each augmentation independently with given probability."""
    if random.random() < p_blur:
        img = apply_gaussian_blur_and_noise(img)
    if random.random() < p_perspective:
        img = apply_perspective_transform(img)
    if random.random() < p_mask:
        img = apply_random_rectangular_mask(img)
    return img


# ──────────────────────────────────────────────────────────────────────
#  LMDB I/O helpers (PARSeq standard format)
# ──────────────────────────────────────────────────────────────────────
#  Keys:  num-samples  (int as UTF-8 string)
#         image-XXXXXXXXX  (JPEG/PNG bytes)
#         label-XXXXXXXXX  (UTF-8 string)

def read_lmdb(lmdb_path):
    """Yield (image_bytes, label_str) from an existing LMDB."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        n = int(txn.get(b'num-samples').decode())
        print(f"  Source LMDB has {n} samples")
        for i in range(1, n + 1):
            idx = f'{i:09d}'
            img_key = f'image-{idx}'.encode()
            lbl_key = f'label-{idx}'.encode()
            img_bytes = txn.get(img_key)
            label = txn.get(lbl_key).decode()
            if img_bytes is not None:
                yield img_bytes, label
    env.close()


def img_to_bytes(img, fmt='PNG'):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_img(b):
    return Image.open(io.BytesIO(b)).convert('RGB')


def write_lmdb(samples, dst_path, map_size_gb=10):
    """Write list of (image_bytes, label) to LMDB."""
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(dst_path), map_size=map_size_gb * (1024 ** 3))
    with env.begin(write=True) as txn:
        for i, (img_bytes, label) in enumerate(samples, 1):
            idx = f'{i:09d}'
            txn.put(f'image-{idx}'.encode(), img_bytes)
            txn.put(f'label-{idx}'.encode(), label.encode())
        txn.put(b'num-samples', str(len(samples)).encode())
    env.close()
    print(f"  Wrote {len(samples)} samples to {dst_path}")


# ──────────────────────────────────────────────────────────────────────
#  Build from crop images + GT JSON  (alternative to LMDB source)
# ──────────────────────────────────────────────────────────────────────

def read_crops_and_gt(crops_dir, gt_json_path):
    """
    Yield (image_bytes, label_str) from a crops folder + ground truth.

    gt_json format:  { "tracklet_id": jersey_number_int, ... }
    crop filenames:  trackletId_frameId.jpg   (e.g. 42_00001234.jpg)
    """
    import json
    with open(gt_json_path) as f:
        gt = json.load(f)

    for fname in sorted(os.listdir(crops_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        # Extract tracklet ID from filename
        tracklet_id = fname.split('_')[0]
        if tracklet_id not in gt:
            continue
        label = gt[tracklet_id]
        # Skip illegible (-1) and soccer balls
        if int(label) < 0:
            continue
        label_str = str(int(label))

        img_path = os.path.join(crops_dir, fname)
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        yield img_bytes, label_str


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Augment training data and build LMDB for PARSeq fine-tuning")

    # Source: either LMDB or crops+GT
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--src_lmdb', help='Path to source LMDB (e.g. data/SoccerNet/lmdb/train)')
    src.add_argument('--crops_dir', help='Path to crop images folder')

    parser.add_argument('--gt_json', help='Path to ground truth JSON (required with --crops_dir)')
    parser.add_argument('--dst_lmdb', required=True,
                        help='Path for output augmented LMDB (e.g. data/SoccerNet/lmdb_augmented/train)')
    parser.add_argument('--num_augmented_copies', type=int, default=5,
                        help='Number of augmented copies per original image (default: 5)')
    parser.add_argument('--p_blur', type=float, default=0.5, help='Prob of blur+noise (default: 0.5)')
    parser.add_argument('--p_perspective', type=float, default=0.5, help='Prob of perspective warp (default: 0.5)')
    parser.add_argument('--p_mask', type=float, default=0.6, help='Prob of occlusion mask (default: 0.6)')
    parser.add_argument('--preview', type=int, default=0,
                        help='Save N preview images to dst_lmdb/preview/ for visual inspection')

    args = parser.parse_args()

    if args.crops_dir and not args.gt_json:
        parser.error("--gt_json is required when using --crops_dir")

    # ── Read source data ──
    print("Reading source data...")
    if args.src_lmdb:
        source_iter = read_lmdb(args.src_lmdb)
    else:
        source_iter = read_crops_and_gt(args.crops_dir, args.gt_json)

    originals = list(source_iter)
    print(f"  Loaded {len(originals)} original samples")

    # ── Generate augmented samples ──
    print(f"Generating {args.num_augmented_copies} augmented copies per sample...")
    all_samples = []

    # Keep originals
    all_samples.extend(originals)

    preview_dir = None
    if args.preview > 0:
        preview_dir = os.path.join(args.dst_lmdb, 'preview')
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
    preview_count = 0

    for i, (img_bytes, label) in enumerate(originals):
        img = bytes_to_img(img_bytes)

        for j in range(args.num_augmented_copies):
            aug_img = augment_image(
                img.copy(),
                p_blur=args.p_blur,
                p_perspective=args.p_perspective,
                p_mask=args.p_mask
            )
            aug_bytes = img_to_bytes(aug_img)
            all_samples.append((aug_bytes, label))

            # Save previews
            if preview_dir and preview_count < args.preview:
                orig_path = os.path.join(preview_dir, f'{i:04d}_label{label}_orig.png')
                aug_path = os.path.join(preview_dir, f'{i:04d}_label{label}_aug{j}.png')
                if not os.path.exists(orig_path):
                    img.save(orig_path)
                aug_img.save(aug_path)
                preview_count += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(originals)} originals...")

    # Shuffle augmented data
    random.shuffle(all_samples)

    total_orig = len(originals)
    total_aug = len(all_samples)
    print(f"\n  Original samples:  {total_orig}")
    print(f"  Total samples:     {total_aug}  ({total_aug / total_orig:.1f}x)")

    # ── Write output LMDB ──
    print(f"\nWriting augmented LMDB to {args.dst_lmdb}...")
    write_lmdb(all_samples, args.dst_lmdb)

    # ── Also create a val split link if it doesn't exist ──
    dst_parent = Path(args.dst_lmdb).parent
    val_dir = dst_parent / 'val'
    if not val_dir.exists():
        print(f"\n  Note: You may also need a val/ LMDB in {dst_parent}")
        print(f"  You can symlink or copy your existing val LMDB there.")

    print("\nDone! To fine-tune PARSeq with augmented data:")
    print(f"  conda run --live-stream -n parseq2 python str/parseq/train.py \\")
    print(f"      +experiment=parseq dataset=real \\")
    print(f"      data.root_dir={dst_parent} \\")
    print(f"      trainer.max_epochs=25 pretrained=parseq \\")
    print(f"      trainer.devices=1 data.batch_size=128 data.max_label_length=2")


if __name__ == '__main__':
    main()
