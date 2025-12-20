import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# -----------------------------
# Settings
# -----------------------------
SOURCE_DIR = "brain_tumor_dataset"
OUTPUT_DIR = "brain_tumor_dataset_fixed"
TARGET_SIZE = 400  # Standardized size (between 225 and 630)
CATEGORIES = ['yes', 'no']

# -----------------------------
# EXPLANATION OF FIXES
# -----------------------------
"""
This script applies multiple preprocessing steps to remove dataset bias:

1. STANDARDIZE SIZE: Resize all images to 400×400
   - Removes the size-based shortcut (630×630 vs 225×225)
   - All images now have identical dimensions

2. CENTER CROP: Focus on the brain region
   - Crops to center 90% of image
   - Removes edge artifacts and borders
   - Forces focus on brain tissue, not corners

3. NORMALIZE BORDERS: Ensure consistent borders
   - Adds small black padding around all images
   - Eliminates border brightness differences

4. QUALITY CHECK: Filters out problematic images
   - Removes images that are too dark/bright
   - Removes images that are too small to process
   - Ensures consistent quality
"""

# -----------------------------
# Preprocessing Functions
# -----------------------------

def center_crop_brain_region(image, crop_percentage=0.9):
    """
    Crop to center region to focus on brain, remove borders
    
    Args:
        image: Input image
        crop_percentage: Percentage of image to keep (0.9 = keep center 90%)
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    # Calculate crop dimensions
    new_h = int(h * crop_percentage)
    new_w = int(w * crop_percentage)
    
    # Calculate crop coordinates (centered)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    
    # Crop
    cropped = image[start_y:start_y+new_h, start_x:start_x+new_w]
    
    return cropped

def add_consistent_border(image, border_size=10, border_color=0):
    """
    Add consistent black border to all images
    
    Args:
        image: Input image
        border_size: Size of border in pixels
        border_color: Color of border (0 = black)
    
    Returns:
        Image with border
    """
    return cv2.copyMakeBorder(
        image, 
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, 
        value=[border_color, border_color, border_color]
    )

def is_valid_image(image):
    """
    Check if image is valid for processing
    
    Returns:
        True if valid, False if should be skipped
    """
    if image is None:
        return False
    
    # Check minimum size
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return False
    
    # Check if image is too dark or too bright (likely corrupt)
    mean_brightness = np.mean(image)
    if mean_brightness < 5 or mean_brightness > 250:
        return False
    
    return True

def preprocess_image(image, target_size=400):
    """
    Apply all preprocessing steps to fix bias
    
    Pipeline:
    1. Validate image
    2. Center crop (remove borders/corners)
    3. Resize to standard size
    4. Add consistent border
    
    Args:
        image: Input image (BGR from cv2.imread)
        target_size: Target size for output
    
    Returns:
        Preprocessed image, or None if invalid
    """
    # Validate
    if not is_valid_image(image):
        return None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 1: Center crop to focus on brain region (removes corners/borders)
    cropped = center_crop_brain_region(image_rgb, crop_percentage=0.85)
    
    # Step 2: Resize to standard size (removes size-based bias)
    resized = cv2.resize(cropped, (target_size, target_size), 
                        interpolation=cv2.INTER_AREA)
    
    # Step 3: Add consistent border (normalizes edge appearance)
    bordered = add_consistent_border(resized, border_size=5, border_color=0)
    
    return bordered

# -----------------------------
# Process Dataset
# -----------------------------

def process_dataset(source_dir, output_dir, target_size=400):
    """
    Process entire dataset with bias-removal preprocessing
    """
    print("="*70)
    print(" "*20 + "DATASET PREPROCESSING")
    print("="*70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}×{target_size}")
    print("="*70)
    print("\nPreprocessing steps:")
    print("  1. Center crop (85% - removes borders)")
    print("  2. Resize to standard size (removes size bias)")
    print("  3. Add consistent borders (normalizes edges)")
    print("  4. Quality checks (filters bad images)")
    print("="*70 + "\n")
    
    # Create output directory
    if os.path.exists(output_dir):
        response = input(f"Output directory '{output_dir}' exists. Overwrite? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    
    # Process each category
    stats = {'total': 0, 'processed': 0, 'skipped': 0}
    
    for category in CATEGORIES:
        print(f"\nProcessing '{category}' category...")
        
        source_path = os.path.join(source_dir, category)
        output_path = os.path.join(output_dir, category)
        os.makedirs(output_path, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(source_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        stats['total'] += len(image_files)
        
        # Process each image with progress bar
        for img_name in tqdm(image_files, desc=f"  {category}"):
            img_path = os.path.join(source_path, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            
            # Preprocess
            processed_img = preprocess_image(img, target_size)
            
            if processed_img is not None:
                # Convert back to BGR for saving
                processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                
                # Save with same filename
                output_img_path = os.path.join(output_path, img_name)
                cv2.imwrite(output_img_path, processed_img_bgr)
                
                stats['processed'] += 1
            else:
                stats['skipped'] += 1
                print(f"    ⚠️  Skipped (invalid): {img_name}")
    
    # Print summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Total images:      {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (invalid): {stats['skipped']}")
    print(f"Success rate:      {stats['processed']/stats['total']*100:.1f}%")
    print("="*70)
    print(f"\n✓ Preprocessed dataset saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Visually inspect a few images in the new folder")
    print("  2. Run the training script with the fixed dataset")
    print("  3. Use Grad-CAM to verify model focuses on brain tissue")
    print("="*70 + "\n")

# -----------------------------
# Visualization Helper
# -----------------------------

def show_before_after_comparison(source_dir, output_dir, num_samples=3):
    """
    Show before/after comparison of preprocessing
    """
    import matplotlib.pyplot as plt
    
    print("\nGenerating before/after comparison...")
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples*4))
    
    for cat_idx, category in enumerate(CATEGORIES):
        source_path = os.path.join(source_dir, category)
        output_path = os.path.join(output_dir, category)
        
        # Get random samples
        files = os.listdir(source_path)
        samples = np.random.choice(files, min(num_samples, len(files)), replace=False)
        
        for idx, filename in enumerate(samples):
            # Load original
            orig_img = cv2.imread(os.path.join(source_path, filename))
            orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Load processed
            proc_img = cv2.imread(os.path.join(output_path, filename))
            proc_img_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            
            row = idx
            col_offset = cat_idx * 2
            
            # Original
            axes[row, col_offset].imshow(orig_img_rgb)
            axes[row, col_offset].set_title(f'{category.upper()} - Original\n{orig_img.shape[1]}×{orig_img.shape[0]}',
                                           fontsize=10, fontweight='bold')
            axes[row, col_offset].axis('off')
            
            # Processed
            axes[row, col_offset + 1].imshow(proc_img_rgb)
            axes[row, col_offset + 1].set_title(f'{category.upper()} - Fixed\n{proc_img.shape[1]}×{proc_img.shape[0]}',
                                               fontsize=10, fontweight='bold',
                                               color='green')
            axes[row, col_offset + 1].axis('off')
    
    plt.suptitle('Before vs After Preprocessing\nNotice: Same size, cropped borders, consistent appearance',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------

def main():
    """Main function"""
    
    print("\n" + "="*70)
    print(" "*15 + "DATASET BIAS REMOVAL TOOL")
    print("="*70)
    print("\nThis script will fix the dataset bias by:")
    print("  ✓ Standardizing all image sizes (removes size bias)")
    print("  ✓ Center cropping (removes corner/border bias)")
    print("  ✓ Adding consistent borders (normalizes edges)")
    print("  ✓ Quality filtering (removes invalid images)")
    print("="*70)
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"\n❌ Error: Source directory '{SOURCE_DIR}' not found!")
        return
    
    # Confirm with user
    print(f"\nThis will create a new dataset folder: '{OUTPUT_DIR}'")
    response = input("Proceed with preprocessing? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Process dataset
    process_dataset(SOURCE_DIR, OUTPUT_DIR, TARGET_SIZE)
    
    # Show comparison
    if os.path.exists(OUTPUT_DIR):
        show_comparison = input("\nShow before/after comparison? (yes/no): ")
        if show_comparison.lower() in ['yes', 'y']:
            show_before_after_comparison(SOURCE_DIR, OUTPUT_DIR, num_samples=3)
    
    print("\n✓ All done! You can now train on the fixed dataset.")
    print(f"  Update DATA_DIR = '{OUTPUT_DIR}' in your training script.\n")

if __name__ == "__main__":
    main()