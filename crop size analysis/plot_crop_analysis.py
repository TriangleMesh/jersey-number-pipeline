import matplotlib.pyplot as plt
import re

# Parse the crop_size_analysis.txt file
heights = []
counts = []

with open('crop_size_analysis.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    
    # Find the height distribution section
    lines = content.split('\n')
    in_distribution = False
    
    for line in lines:
        # Look for lines with height information: "  Height XXpx:    YYY images ( Z.ZZ%)"
        if 'Height' in line and 'px:' in line:
            in_distribution = True
            # Extract height and count
            match = re.search(r'(\d+)px:\s+(\d+)\s+images', line)
            if match:
                height = int(match.group(1))
                count = int(match.group(2))
                heights.append(height)
                counts.append(count)

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Height Distribution Line Chart
ax1.plot(heights, counts, marker='o', linewidth=2, markersize=4, color='#2E86AB')
ax1.fill_between(heights, counts, alpha=0.3, color='#2E86AB')
ax1.set_xlabel('Crop Height (pixels)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax1.set_title('Jersey Number Crop Height Distribution', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(min(heights)-2, max(heights)+2)

# Add annotation for threshold
ax1.axvline(x=64, color='red', linestyle='--', linewidth=2, label='Upsampling Threshold (64px)')
ax1.legend(fontsize=11)

# Plot 2: Statistics
stats_text = f"""
Total Images: 87,319
Images < 64px: 82,060 (93.98%)
Images ≥ 64px: 5,259 (6.02%)

Height Range: {min(heights)}px - {max(heights)}px
Peak Height: {heights[counts.index(max(counts))]}px (with {max(counts)} images)
Most Common Range: 38-44px

Key Insight:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
94% of cropped jersey numbers are smaller than 
64 pixels in height, requiring 4x upsampling to 
achieve 160+ pixels for better recognition accuracy.
"""

ax2.text(0.5, 0.5, stats_text, transform=ax2.transAxes,
         fontsize=11, verticalalignment='center', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         family='monospace')
ax2.axis('off')

plt.tight_layout()
plt.savefig('crop_height_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved as 'crop_height_distribution.png'")
print(f"\nSummary:")
print(f"  Total data points: {len(heights)}")
print(f"  Height range: {min(heights)}-{max(heights)} pixels")
print(f"  Peak: {heights[counts.index(max(counts))]}px with {max(counts)} images")
print(f"  Images < 64px: {sum(1 for h in heights if h < 64)} height categories containing {sum(counts[i] for i in range(len(heights)) if heights[i] < 64)} images")

plt.show()
