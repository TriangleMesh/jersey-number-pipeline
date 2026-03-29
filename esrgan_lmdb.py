#!/usr/bin/env python3
"""
Apply Real-ESRGAN to all images in an LMDB dataset and write a new LMDB.

Usage:
    python esrgan_lmdb.py \
        --src_root data/SoccerNet/lmdb \
        --dst_root data/SoccerNet/lmdb_esrgan \
        --model_path weights/RealESRGAN_x4plus.pth \
        [--half]   # use half precision on GPU for speed

This script walks the src_root directory tree, finds every LMDB (data.mdb),
applies Real-ESRGAN to each stored image, and writes a new LMDB with the
same structure and labels at the corresponding path under dst_root.

Run this ONCE before re-fine-tuning PARSeq. After running, update
configuration.py and retrain with:
    python main.py SoccerNet train --train_str
"""

import argparse
import io
import os

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm


COMMIT_INTERVAL = 500  # commit lmdb transaction every N images


def load_esrgan(model_path: str, scale: int = 4, half: bool = False):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=scale,
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=128,       # tile processing avoids OOM on large images
        tile_pad=10,
        pre_pad=0,
        half=half,
    )
    return upsampler


def process_lmdb(src_path: str, dst_path: str, upsampler) -> None:
    """Read src LMDB, apply ESRGAN to every image, write to dst LMDB."""
    os.makedirs(dst_path, exist_ok=True)

    src_env = lmdb.open(src_path, readonly=True, lock=False, readahead=False)
    with src_env.begin() as txn:
        raw = txn.get(b'num-samples')
        if raw is None:
            print(f"  [skip] no num-samples key in {src_path}")
            src_env.close()
            return
        num_samples = int(raw)

    # ESRGAN 4x → each image is ~16x larger in bytes; add a large buffer
    map_size = max(src_env.info()['map_size'] * 25, 1 << 34)  # at least 16 GB
    dst_env = lmdb.open(dst_path, map_size=map_size)

    with src_env.begin() as src_txn:
        dst_txn = dst_env.begin(write=True)
        for idx in tqdm(range(1, num_samples + 1), desc=f'  {os.path.basename(src_path)}'):
            img_key = f'image-{idx:09d}'.encode()
            lbl_key = f'label-{idx:09d}'.encode()

            img_bytes = src_txn.get(img_key)
            label     = src_txn.get(lbl_key)

            if img_bytes is None:
                continue

            # Decode
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception as e:
                print(f"  decode error idx={idx}: {e}; skipping")
                continue

            # Apply Real-ESRGAN (expects BGR numpy)
            bgr = np.array(pil_img)[:, :, ::-1]
            try:
                output, _ = upsampler.enhance(bgr, outscale=4)
                result_pil = Image.fromarray(output[:, :, ::-1])  # BGR→RGB
            except Exception as e:
                print(f"  ESRGAN error idx={idx}: {e}; using original")
                result_pil = pil_img

            # Encode to PNG bytes
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')

            dst_txn.put(img_key, buf.getvalue())
            dst_txn.put(lbl_key, label)

            # Commit periodically to avoid a single huge transaction
            if idx % COMMIT_INTERVAL == 0:
                dst_txn.commit()
                dst_txn = dst_env.begin(write=True)

        dst_txn.put(b'num-samples', str(num_samples).encode())
        dst_txn.commit()

    src_env.close()
    dst_env.close()
    print(f"  Done: {num_samples} images → {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Apply Real-ESRGAN to every image in an LMDB dataset tree.'
    )
    parser.add_argument('--src_root', required=True,
                        help='Source LMDB root, e.g. data/SoccerNet/lmdb')
    parser.add_argument('--dst_root', required=True,
                        help='Output LMDB root, e.g. data/SoccerNet/lmdb_esrgan')
    parser.add_argument('--model_path', default='weights/RealESRGAN_x4plus.pth',
                        help='Path to Real-ESRGAN weights file')
    parser.add_argument('--half', action='store_true', default=False,
                        help='Use FP16 (faster on GPU, may cause issues on CPU)')
    args = parser.parse_args()

    print(f"Loading Real-ESRGAN from {args.model_path} ...")
    upsampler = load_esrgan(args.model_path, scale=4, half=args.half)
    print("Model loaded.\n")

    found = 0
    for dirpath, _, filenames in os.walk(args.src_root):
        if 'data.mdb' in filenames:
            found += 1
            rel = os.path.relpath(dirpath, args.src_root)
            dst = os.path.join(args.dst_root, rel)
            print(f"Processing: {dirpath}")
            process_lmdb(dirpath, dst, upsampler)
            print()

    if found == 0:
        print(f"No LMDB databases found under {args.src_root}")
    else:
        print(f"All done. {found} LMDB(s) processed.")
        print(f"New LMDBs are at: {args.dst_root}")
        print("\nNext steps:")
        print("  1. python main.py SoccerNet train --train_str   (re-fine-tune PARSeq)")
        print("  2. python main.py SoccerNet test                (run pipeline with ESRGAN inference)")


if __name__ == '__main__':
    main()
