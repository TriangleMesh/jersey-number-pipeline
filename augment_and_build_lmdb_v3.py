"""
augment_and_build_lmdb_v3.py
─────────────────────────────
Minimal, literature-backed augmentation for PARSeq fine-tuning.

PARSeq already applies RandAugment internally during training (N=3, M=5)
with: AutoContrast, Equalize, Invert, Rotate, Posterize, Solarize, Color,
Contrast, Brightness, ShearX, ShearY, TranslateX, TranslateY, GaussianBlur,
PoissonNoise.

This script adds ONLY the domain-specific augmentations that PARSeq lacks
and that the STR literature identifies as gaps for broadcast sports video:

  1. MotionBlur  — sports video has motion blur, not just Gaussian
  2. Pixelate    — simulates low-resolution broadcast crops
  3. JPEG compression — broadcast compression artifacts

Each augmentation is applied independently with moderate probability.

Based on:
  - STRAug (Atienza, ICCV Workshop 2021) — Blur and Camera groups
  - Koshkina & Elder (CVPR Workshops 2024) — error analysis
  - Lébl et al. (SCIA 2023) — blur-type matching is essential

Usage:
  python augment_and_build_lmdb_v3.py ^
      --crops_dir  out/SoccerNetResults/crops_train/imgs ^
      --gt_json    data/SoccerNet/train/train_gt.json ^
      --dst_lmdb   data/SoccerNet/lmdb_augmented_v3/train ^
      --num_augmented_copies 5 ^
      --preview 20
"""

import argparse
import io
import os
import random
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image, ImageFilter


# ──────────────────────────────────────────────────────────────────────
#  Three domain-specific augmentations (not in PARSeq's default pipeline)
# ──────────────────────────────────────────────────────────────────────

def apply_motion_blur(img):
    """Simulates camera/player movement in broadcast sports video.
    Uses horizontal or vertical directional blur."""
    kernel_size = 3
    direction = random.choice(['horizontal', 'vertical', 'diagonal'])

    if direction == 'horizontal':
        kernel = [0] * (kernel_size * kernel_size)
        mid = kernel_size // 2
        for i in range(kernel_size):
            kernel[mid * kernel_size + i] = 1.0
    elif direction == 'vertical':
        kernel = [0] * (kernel_size * kernel_size)
        mid = kernel_size // 2
        for i in range(kernel_size):
            kernel[i * kernel_size + mid] = 1.0
    else:  # diagonal
        kernel = [0] * (kernel_size * kernel_size)
        for i in range(kernel_size):
            kernel[i * kernel_size + i] = 1.0

    total = sum(kernel)
    return img.filter(ImageFilter.Kernel(
        size=(kernel_size, kernel_size),
        kernel=kernel,
        scale=total,
        offset=0
    ))


def apply_pixelate(img):
    """Simulates low-resolution broadcast crops.
    Downsamples then upsamples back to original size."""
    w, h = img.size
    factor = 2  # 2x downscale
    small = img.resize((max(1, w // factor), max(1, h // factor)), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def apply_jpeg_compress(img):
    """Simulates broadcast JPEG/video compression artifacts."""
    quality = random.randint(30, 70)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def augment_image(img, p_motion=0.3, p_pixelate=0.2, p_jpeg=0.3):
    """Apply at most one spatial degradation + optional JPEG compression."""
    roll = random.random()
    if roll < p_motion:
        img = apply_motion_blur(img)
    elif roll < p_motion + p_pixelate:
        img = apply_pixelate(img)
    # JPEG can still stack (it's mild)
    if random.random() < p_jpeg:
        img = apply_jpeg_compress(img)
    return img


# ──────────────────────────────────────────────────────────────────────
#  LMDB I/O
# ──────────────────────────────────────────────────────────────────────

def read_lmdb(lmdb_path):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        n = int(txn.get(b'num-samples').decode())
        print(f"  Source LMDB has {n} samples")
        for i in range(1, n + 1):
            idx = f'{i:09d}'
            img_bytes = txn.get(f'image-{idx}'.encode())
            label = txn.get(f'label-{idx}'.encode()).decode()
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


def read_crops_and_gt(crops_dir, gt_json_path):
    import json
    with open(gt_json_path) as f:
        gt = json.load(f)

    for fname in sorted(os.listdir(crops_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        tracklet_id = fname.split('_')[0]
        if tracklet_id not in gt:
            continue
        label = gt[tracklet_id]
        if int(label) < 0:
            continue

        img_path = os.path.join(crops_dir, fname)
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        yield img_bytes, str(int(label))


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Domain-specific augmentation for PARSeq fine-tuning on sports broadcast crops")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--src_lmdb', help='Path to source LMDB')
    src.add_argument('--crops_dir', help='Path to crop images folder')

    parser.add_argument('--gt_json', help='Ground truth JSON (required with --crops_dir)')
    parser.add_argument('--dst_lmdb', required=True, help='Output LMDB path')
    parser.add_argument('--num_augmented_copies', type=int, default=5,
                        help='Augmented copies per original (default: 5)')
    parser.add_argument('--p_motion', type=float, default=0.4,
                        help='Probability of motion blur (default: 0.4)')
    parser.add_argument('--p_pixelate', type=float, default=0.3,
                        help='Probability of pixelation (default: 0.3)')
    parser.add_argument('--p_jpeg', type=float, default=0.3,
                        help='Probability of JPEG compression (default: 0.3)')
    parser.add_argument('--preview', type=int, default=0,
                        help='Save N preview images')

    args = parser.parse_args()

    if args.crops_dir and not args.gt_json:
        parser.error("--gt_json is required when using --crops_dir")

    print("Reading source data...")
    if args.src_lmdb:
        source_iter = read_lmdb(args.src_lmdb)
    else:
        source_iter = read_crops_and_gt(args.crops_dir, args.gt_json)

    originals = list(source_iter)
    print(f"  Loaded {len(originals)} original samples")

    print(f"Generating {args.num_augmented_copies} augmented copies per sample...")
    print(f"  Augmentations: MotionBlur(p={args.p_motion}), "
          f"Pixelate(p={args.p_pixelate}), JPEG(p={args.p_jpeg})")
    all_samples = list(originals)  # keep originals

    preview_dir = None
    if args.preview > 0:
        preview_dir = os.path.join(args.dst_lmdb, 'preview')
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
    preview_count = 0

    for i, (img_bytes, label) in enumerate(originals):
        img = bytes_to_img(img_bytes)

        for j in range(args.num_augmented_copies):
            aug_img = augment_image(img.copy(),
                                    p_motion=args.p_motion,
                                    p_pixelate=args.p_pixelate,
                                    p_jpeg=args.p_jpeg)
            all_samples.append((img_to_bytes(aug_img), label))

            if preview_dir and preview_count < args.preview:
                orig_path = os.path.join(preview_dir, f'{i:04d}_label{label}_orig.png')
                aug_path = os.path.join(preview_dir, f'{i:04d}_label{label}_aug{j}.png')
                if not os.path.exists(orig_path):
                    img.save(orig_path)
                aug_img.save(aug_path)
                preview_count += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(originals)}...")

    random.shuffle(all_samples)
    print(f"\n  Original: {len(originals)}, Total: {len(all_samples)} ({len(all_samples)/len(originals):.1f}x)")

    print(f"\nWriting LMDB to {args.dst_lmdb}...")
    write_lmdb(all_samples, args.dst_lmdb)

    dst_parent = Path(args.dst_lmdb).parent
    if not (dst_parent / 'val').exists():
        print(f"\n  Remember to copy val LMDB to {dst_parent / 'val'}")


if __name__ == '__main__':
    main()
