#!/usr/bin/env python3
import os
import subprocess
from collections import defaultdict

img_dir = "SoccerNetResults/crops/imgs"
heights = defaultdict(int)
count = 0
small_count = 0
small_files = []

print("Checking image dimensions...")

for idx, img_file in enumerate(sorted(os.listdir(img_dir))):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(img_dir, img_file)
        try:
            # Use sips command to get image height
            result = subprocess.run(['sips', '-g', 'pixelHeight', img_path], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                output = result.stdout.strip()
                height_str = output.split(': ')[-1]
                height = int(height_str)
                
                heights[height] += 1
                
                if height < 64:
                    small_count += 1
                    small_files.append((img_file, height))
                count += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"Checked {idx + 1} images...")
        except Exception as e:
            print(f"Error: {img_file}: {e}")

print(f"\n" + "=" * 60)
print(f"Total images: {count}")
print(f"Images < 64px: {small_count}")
print(f"Percentage: {100 * small_count / count:.2f}%")

print(f"\nHeight Distribution Statistics:")
for h in sorted(heights.keys()):
    pct = 100 * heights[h] / count
    print(f"  Height {h:3d}px: {heights[h]:4d} images ({pct:5.2f}%)")

if small_files:
    print(f"\nExample images < 64px (first 20):")
    for fname, h in small_files[:20]:
        print(f"  {fname}: {h}px")
