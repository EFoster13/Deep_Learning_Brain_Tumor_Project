import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# -----------------------------
# Settings
# -----------------------------
DATA_DIR = "brain_tumor_dataset"
CATEGORIES = ['yes', 'no']

# -----------------------------
# Load All Images
# -----------------------------
def load_all_images(data_path):
    """Load all images from both categories"""
    images_data = {'yes': [], 'no': []}
    
    print("Loading images...")
    for category in CATEGORIES:
        path = os.path.join(data_path, category)
        count = 0
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images_data[category].append({
                    'image': img_rgb,
                    'filename': img_name,
                    'shape': img_rgb.shape
                })
                count += 1
        print(f"✓ Loaded {count} images from '{category}' folder")
    
    return images_data

# -----------------------------
# Analysis 1: Image Dimensions
# -----------------------------
def analyze_dimensions(images_data):
    """Analyze if image dimensions differ between classes"""
    print("\n" + "="*70)
    print("ANALYSIS 1: IMAGE DIMENSIONS")
    print("="*70)
    
    for category in CATEGORIES:
        shapes = [img['shape'] for img in images_data[category]]
        heights = [s[0] for s in shapes]
        widths = [s[1] for s in shapes]
        
        print(f"\n{category.upper()} class:")
        print(f"  Height range: {min(heights)} - {max(heights)} pixels")
        print(f"  Width range:  {min(widths)} - {max(widths)} pixels")
        print(f"  Most common shape: {max(set(shapes), key=shapes.count)}")
    
    # Visualize dimension distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, category in enumerate(CATEGORIES):
        shapes = [img['shape'] for img in images_data[category]]
        heights = [s[0] for s in shapes]
        widths = [s[1] for s in shapes]
        
        axes[idx].scatter(widths, heights, alpha=0.6)
        axes[idx].set_xlabel('Width (pixels)', fontsize=12)
        axes[idx].set_ylabel('Height (pixels)', fontsize=12)
        axes[idx].set_title(f'{category.upper()} - Image Dimensions', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# -----------------------------
# Analysis 2: Corner Analysis
# -----------------------------
def analyze_corners(images_data):
    """Analyze corner pixels - THIS IS WHERE THE BIAS LIKELY IS"""
    print("\n" + "="*70)
    print("ANALYSIS 2: CORNER PIXEL ANALYSIS")
    print("="*70)
    print("Examining if corners have different patterns between classes...")
    
    corner_stats = {'yes': [], 'no': []}
    
    # For each category, analyze corner regions
    for category in CATEGORIES:
        for img_data in images_data[category]:
            img = img_data['image']
            h, w = img.shape[:2]
            
            # Define corner regions (10% of image size)
            corner_size_h = h // 10
            corner_size_w = w // 10
            
            # Extract four corners
            top_left = img[:corner_size_h, :corner_size_w]
            top_right = img[:corner_size_h, -corner_size_w:]
            bottom_left = img[-corner_size_h:, :corner_size_w]
            bottom_right = img[-corner_size_h:, -corner_size_w:]
            
            # Calculate average brightness of each corner
            corners = [top_left, top_right, bottom_left, bottom_right]
            avg_brightness = np.mean([np.mean(corner) for corner in corners])
            
            corner_stats[category].append(avg_brightness)
    
    # Statistical comparison
    yes_mean = np.mean(corner_stats['yes'])
    no_mean = np.mean(corner_stats['no'])
    yes_std = np.std(corner_stats['yes'])
    no_std = np.std(corner_stats['no'])
    
    print(f"\nCorner Brightness Statistics:")
    print(f"  YES (tumor):   Mean = {yes_mean:.2f}, Std = {yes_std:.2f}")
    print(f"  NO (no tumor): Mean = {no_mean:.2f}, Std = {no_std:.2f}")
    print(f"  Difference:    {abs(yes_mean - no_mean):.2f} intensity units")
    
    if abs(yes_mean - no_mean) > 20:
        print("\n⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
        print("    The corners have very different brightness between classes.")
        print("    This is likely what the model is exploiting!")
    
    # Visualize corner brightness distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(corner_stats['yes'], bins=30, alpha=0.7, label='YES (tumor)', color='red', edgecolor='black')
    plt.hist(corner_stats['no'], bins=30, alpha=0.7, label='NO (no tumor)', color='green', edgecolor='black')
    plt.xlabel('Average Corner Brightness', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Corner Brightness Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    box_data = [corner_stats['yes'], corner_stats['no']]
    plt.boxplot(box_data, labels=['YES (tumor)', 'NO (no tumor)'])
    plt.ylabel('Average Corner Brightness', fontsize=12)
    plt.title('Corner Brightness Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# -----------------------------
# Analysis 3: Average Images
# -----------------------------
def analyze_average_images(images_data):
    """Create average image for each class to see overall patterns"""
    print("\n" + "="*70)
    print("ANALYSIS 3: AVERAGE IMAGES")
    print("="*70)
    print("Creating average images for each class...")
    
    avg_images = {}
    
    for category in CATEGORIES:
        # Resize all images to a common size for averaging
        common_size = (300, 300)
        resized_images = []
        
        for img_data in images_data[category]:
            img_resized = cv2.resize(img_data['image'], common_size)
            resized_images.append(img_resized.astype(np.float32))
        
        # Calculate average
        avg_img = np.mean(resized_images, axis=0).astype(np.uint8)
        avg_images[category] = avg_img
    
    # Visualize average images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(avg_images['yes'])
    axes[0].set_title('AVERAGE: YES (Tumor)', fontsize=14, fontweight='bold', color='red')
    axes[0].axis('off')
    
    axes[1].imshow(avg_images['no'])
    axes[1].set_title('AVERAGE: NO (No Tumor)', fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # Difference image
    diff = np.abs(avg_images['yes'].astype(np.float32) - avg_images['no'].astype(np.float32))
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
    
    axes[2].imshow(diff_normalized)
    axes[2].set_title('DIFFERENCE MAP\n(Bright = Different)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Average Images Show Class-Level Patterns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Look at the difference map (right image):")
    print("  - Bright areas = regions that differ between classes")
    print("  - If corners are bright, that's the artifact!")

# -----------------------------
# Analysis 4: Edge and Border Analysis
# -----------------------------
def analyze_edges(images_data):
    """Analyze if image borders/edges differ between classes"""
    print("\n" + "="*70)
    print("ANALYSIS 4: BORDER ANALYSIS")
    print("="*70)
    print("Checking if image borders differ between classes...")
    
    border_stats = {'yes': {'top': [], 'bottom': [], 'left': [], 'right': []},
                   'no': {'top': [], 'bottom': [], 'left': [], 'right': []}}
    
    for category in CATEGORIES:
        for img_data in images_data[category]:
            img = img_data['image']
            h, w = img.shape[:2]
            
            border_width = 5  # pixels
            
            # Extract borders
            top_border = img[:border_width, :]
            bottom_border = img[-border_width:, :]
            left_border = img[:, :border_width]
            right_border = img[:, -border_width:]
            
            # Calculate average brightness
            border_stats[category]['top'].append(np.mean(top_border))
            border_stats[category]['bottom'].append(np.mean(bottom_border))
            border_stats[category]['left'].append(np.mean(left_border))
            border_stats[category]['right'].append(np.mean(right_border))
    
    # Compare borders
    print("\nBorder Brightness Comparison:")
    for position in ['top', 'bottom', 'left', 'right']:
        yes_mean = np.mean(border_stats['yes'][position])
        no_mean = np.mean(border_stats['no'][position])
        diff = abs(yes_mean - no_mean)
        
        print(f"  {position.upper():8} - YES: {yes_mean:.1f}, NO: {no_mean:.1f}, Diff: {diff:.1f}")
        
        if diff > 20:
            print(f"           ⚠️  SIGNIFICANT DIFFERENCE in {position} border!")
    
    # Visualize border comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    positions = ['top', 'bottom', 'left', 'right']
    
    for idx, position in enumerate(positions):
        ax = axes[idx // 2, idx % 2]
        ax.hist(border_stats['yes'][position], bins=20, alpha=0.7, 
               label='YES (tumor)', color='red', edgecolor='black')
        ax.hist(border_stats['no'][position], bins=20, alpha=0.7, 
               label='NO (no tumor)', color='green', edgecolor='black')
        ax.set_xlabel('Border Brightness', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{position.upper()} Border Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# -----------------------------
# Analysis 5: Sample Visualization
# -----------------------------
def visualize_samples(images_data):
    """Show sample images with corner regions highlighted"""
    print("\n" + "="*70)
    print("ANALYSIS 5: SAMPLE IMAGES WITH CORNER HIGHLIGHTING")
    print("="*70)
    
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    
    for cat_idx, category in enumerate(CATEGORIES):
        # Select 6 random samples
        samples = np.random.choice(len(images_data[category]), 
                                  min(6, len(images_data[category])), 
                                  replace=False)
        
        for idx, sample_idx in enumerate(samples):
            img = images_data[category][sample_idx]['image'].copy()
            h, w = img.shape[:2]
            
            # Draw rectangles around corners
            corner_size_h = h // 10
            corner_size_w = w // 10
            
            # Top-left
            cv2.rectangle(img, (0, 0), (corner_size_w, corner_size_h), (255, 0, 0), 3)
            # Top-right
            cv2.rectangle(img, (w-corner_size_w, 0), (w, corner_size_h), (255, 0, 0), 3)
            # Bottom-left
            cv2.rectangle(img, (0, h-corner_size_h), (corner_size_w, h), (255, 0, 0), 3)
            # Bottom-right
            cv2.rectangle(img, (w-corner_size_w, h-corner_size_h), (w, h), (255, 0, 0), 3)
            
            axes[cat_idx, idx].imshow(img)
            axes[cat_idx, idx].set_title(f"{category.upper()}", 
                                        fontsize=11, fontweight='bold',
                                        color='red' if category == 'yes' else 'green')
            axes[cat_idx, idx].axis('off')
    
    plt.suptitle('Sample Images with Corner Regions Highlighted (Red Boxes)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Red boxes show the corner regions the model might be focusing on")
    print("  Compare the corners between YES and NO classes")

# -----------------------------
# Summary Report
# -----------------------------
def generate_summary_report(images_data):
    """Generate final diagnostic summary"""
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY REPORT")
    print("="*70)
    
    yes_count = len(images_data['yes'])
    no_count = len(images_data['no'])
    
    print(f"\nDataset Composition:")
    print(f"  YES (tumor):   {yes_count} images")
    print(f"  NO (no tumor): {no_count} images")
    print(f"  Class balance: {yes_count/(yes_count+no_count)*100:.1f}% vs {no_count/(yes_count+no_count)*100:.1f}%")
    
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   Based on the visualizations above, look for:")
    print(f"   1. Different corner brightness between classes")
    print(f"   2. Different border patterns")
    print(f"   3. Bright corners in the difference map")
    print(f"   4. Consistent artifacts in one class but not the other")
    
    print(f"\n💡 RECOMMENDED ACTIONS:")
    print(f"   If significant differences found:")
    print(f"   1. Crop images to remove corners/borders")
    print(f"   2. Apply center cropping during augmentation")
    print(f"   3. Use circular masks to focus on brain region")
    print(f"   4. Standardize preprocessing for both classes")
    print(f"   5. Consider finding a better-balanced dataset")
    
    print("\n" + "="*70)

# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Run all diagnostic analyses"""
    
    print("="*70)
    print(" "*20 + "DATASET DIAGNOSTIC TOOL")
    print("="*70)
    print("This tool will analyze your dataset to find potential biases")
    print("that the model might be exploiting instead of learning real features.")
    print("="*70)
    
    # Load all images
    images_data = load_all_images(DATA_DIR)
    
    # Run all analyses
    analyze_dimensions(images_data)
    
    input("\nPress Enter to continue to corner analysis...")
    analyze_corners(images_data)
    
    input("\nPress Enter to continue to average images...")
    analyze_average_images(images_data)
    
    input("\nPress Enter to continue to border analysis...")
    analyze_edges(images_data)
    
    input("\nPress Enter to see sample visualizations...")
    visualize_samples(images_data)
    
    # Generate final report
    generate_summary_report(images_data)
    
    print("\n✓ Diagnostic analysis complete!")
    print("Review all visualizations to understand the dataset bias.")

if __name__ == "__main__":
    main()