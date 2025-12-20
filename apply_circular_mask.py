import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

# -----------------------------
# Settings
# -----------------------------
SOURCE_DIR = "brain_tumor_dataset"
OUTPUT_DIR = "brain_tumor_dataset_masked"
TARGET_SIZE = 400
CATEGORIES = ['yes', 'no']

# -----------------------------
# CIRCULAR MASK EXPLANATION
# -----------------------------
"""
This preprocessing creates a CIRCULAR MASK for brain scans:

WHAT IT DOES:
1. Resize image to standard size
2. Create circular mask centered on image
3. Black out everything outside the circle
4. Result: Perfect circle with brain inside, black corners

"""

# -----------------------------
# Create Circular Mask
# -----------------------------
def create_circular_mask(h, w, center=None, radius=None):
    """
    Create a circular boolean mask
    
    Args:
        h: Height of image
        w: Width of image
        center: (x, y) center of circle. If None, uses image center
        radius: Radius of circle. If None, uses 90% of smallest dimension
    
    Returns:
        Boolean mask (True inside circle, False outside)
    """
    if center is None:
        center = (w // 2, h // 2)
    
    if radius is None:
        # Use 90% of the smallest dimension as radius
        # This ensures the entire brain fits while maximizing brain area
        radius = int(min(h, w) * 0.45)  # 0.45 because diameter = 0.9
    
    # Create coordinate arrays
    Y, X = np.ogrid[:h, :w]
    
    # Calculate distance from center for each pixel
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Create mask (True where distance <= radius)
    mask = dist_from_center <= radius
    
    return mask

def apply_circular_mask_to_image(image, mask_radius_ratio=0.90):
    """
    Apply circular mask to an image, blacking out corners
    
    Args:
        image: Input image (RGB)
        mask_radius_ratio: Ratio of image size to use for mask (0.90 = 90%)
    
    Returns:
        Masked image with black corners
    """
    h, w = image.shape[:2]
    
    # Create circular mask
    center = (w // 2, h // 2)
    radius = int(min(h, w) * mask_radius_ratio / 2)
    
    mask = create_circular_mask(h, w, center, radius)
    
    # Create output image (start with black)
    masked_image = np.zeros_like(image)
    
    # Copy only the pixels inside the circle
    masked_image[mask] = image[mask]
    
    return masked_image

# -----------------------------
# Enhanced Preprocessing Pipeline
# -----------------------------
def preprocess_with_circular_mask(image, target_size=400):
    """
    Complete preprocessing pipeline with circular mask
    
    Pipeline:
    1. Validate image
    2. Convert to RGB
    3. Resize to target size
    4. Apply circular mask (BLACK OUT CORNERS!)
    5. Result: Clean circular brain scan
    
    Args:
        image: Input image (BGR from cv2)
        target_size: Target size for output
    
    Returns:
        Preprocessed image with circular mask, or None if invalid
    """
    # Validate
    if image is None:
        return None
    
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return None
    
    # Check brightness
    mean_brightness = np.mean(image)
    if mean_brightness < 5 or mean_brightness > 250:
        return None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to standard size
    resized = cv2.resize(image_rgb, (target_size, target_size), 
                        interpolation=cv2.INTER_AREA)
    
    # Apply circular mask - THIS IS THE KEY STEP!
    masked = apply_circular_mask_to_image(resized, mask_radius_ratio=0.90)
    
    return masked

# -----------------------------
# Visualization
# -----------------------------
def show_mask_demonstration():
    """Show what the circular mask does"""
    print("\n" + "="*70)
    print("CIRCULAR MASK DEMONSTRATION")
    print("="*70)
    
    # Create a sample gradient image
    sample = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(400):
        for j in range(400):
            sample[i, j] = [i*255//400, j*255//400, 128]
    
    # Apply mask
    masked = apply_circular_mask_to_image(sample, mask_radius_ratio=0.90)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(sample)
    axes[0].set_title('Before: Full Image\n(Corners visible)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show just the mask
    h, w = sample.shape[:2]
    mask = create_circular_mask(h, w, radius=int(min(h,w)*0.45))
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Circular Mask\n(White=Keep, Black=Remove)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(masked)
    axes[2].set_title('After: Masked Image\n(Corners BLACK ✓)', fontsize=12, fontweight='bold', color='green')
    axes[2].axis('off')
    
    plt.suptitle('How Circular Masking Works', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✓ Notice: Corners are now PURE BLACK")
    print("✓ Only the circular center region remains")
    print("✓ Model CANNOT use corners as features anymore!\n")

# -----------------------------
# Process Dataset
# -----------------------------
def process_dataset_with_masks(source_dir, output_dir, target_size=400):
    """Process entire dataset with circular masks"""
    
    print("="*70)
    print(" "*15 + "CIRCULAR MASK PREPROCESSING")
    print("="*70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}×{target_size}")
    print(f"Mask: Circular (90% diameter)")
    print("="*70)
    print("\n🎯 KEY FEATURE: Corners will be PURE BLACK")
    print("   Model cannot use corners/edges as shortcuts!")
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
        
        # Get image files
        image_files = [f for f in os.listdir(source_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        stats['total'] += len(image_files)
        
        # Process with progress bar
        for img_name in tqdm(image_files, desc=f"  {category}"):
            img_path = os.path.join(source_path, img_name)
            
            # Read
            img = cv2.imread(img_path)
            
            # Preprocess with circular mask
            processed = preprocess_with_circular_mask(img, target_size)
            
            if processed is not None:
                # Convert back to BGR for saving
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                
                # Save
                output_img_path = os.path.join(output_path, img_name)
                cv2.imwrite(output_img_path, processed_bgr)
                
                stats['processed'] += 1
            else:
                stats['skipped'] += 1
                print(f"    ⚠️  Skipped: {img_name}")
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Total images:           {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (invalid):     {stats['skipped']}")
    print(f"Success rate:          {stats['processed']/stats['total']*100:.1f}%")
    print("="*70)
    print(f"\n✓ Masked dataset saved to: {output_dir}")
    print("\nALL CORNERS ARE NOW PURE BLACK")
    print("   The model MUST focus on brain tissue now!")
    print("\nNext steps:")
    print("  1. Inspect some images in the new folder")
    print("  2. Update training script: DATA_DIR = 'brain_tumor_dataset_masked'")
    print("  3. Retrain the model")
    print("  4. Use Grad-CAM to verify it focuses on brain tissue")
    print("="*70 + "\n")

# -----------------------------
# Before/After Comparison
# -----------------------------
def show_before_after_comparison(source_dir, output_dir, num_samples=4):
    """Show before/after comparison"""
    
    print("\nGenerating before/after comparison...")
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples*4))
    
    for cat_idx, category in enumerate(CATEGORIES):
        source_path = os.path.join(source_dir, category)
        output_path = os.path.join(output_dir, category)
        
        # Get samples
        files = os.listdir(source_path)
        samples = np.random.choice(files, min(num_samples//2, len(files)), replace=False)
        
        for idx, filename in enumerate(samples):
            row = cat_idx * (num_samples // 2) + idx
            
            # Load original
            orig = cv2.imread(os.path.join(source_path, filename))
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            
            # Load masked
            masked = cv2.imread(os.path.join(output_path, filename))
            masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
            
            # Original
            axes[row, 0].imshow(orig_rgb)
            axes[row, 0].set_title(f'{category.upper()} - Original', fontsize=10, fontweight='bold')
            axes[row, 0].axis('off')
            
            # Original with overlay showing what will be masked
            overlay = orig_rgb.copy()
            h, w = overlay.shape[:2]
            mask = create_circular_mask(h, w, radius=int(min(h,w)*0.45))
            overlay[~mask] = overlay[~mask] // 3  # Darken areas that will be masked
            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title('What Gets Masked\n(Darkened=Removed)', fontsize=10, fontweight='bold')
            axes[row, 1].axis('off')
            
            # Masked result
            axes[row, 2].imshow(masked_rgb)
            axes[row, 2].set_title('After Circular Mask\n(Black corners ✓)', fontsize=10, fontweight='bold', color='green')
            axes[row, 2].axis('off')
            
            # Zoom on corners to show they're black
            corner = masked_rgb[:100, :100]  # Top-left corner
            axes[row, 3].imshow(corner)
            axes[row, 3].set_title('Corner Zoom\n(Pure black!)', fontsize=10, fontweight='bold', color='green')
            axes[row, 3].axis('off')
    
    plt.suptitle('Before vs After: Circular Mask Removes ALL Corner Information',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------
def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print(" "*10 + "CIRCULAR MASK DATASET PREPROCESSOR")
    print("="*70)
    print("\nThis will create a new dataset where:")
    print("  ✓ All images are perfectly circular")
    print("  ✓ ALL corners are pure black (RGB: 0,0,0)")
    print("  ✓ Model CANNOT use corner/edge shortcuts")
    print("  ✓ Forces focus on brain tissue only")
    print("="*70)
    
    # Check source
    if not os.path.exists(SOURCE_DIR):
        print(f"\n❌ Error: Source directory '{SOURCE_DIR}' not found!")
        return
    
    # Show demonstration
    show_demo = input("\nShow circular mask demonstration? (yes/no): ")
    if show_demo.lower() in ['yes', 'y']:
        show_mask_demonstration()
    
    # Confirm processing
    response = input(f"\nProceed with processing? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Process
    process_dataset_with_masks(SOURCE_DIR, OUTPUT_DIR, TARGET_SIZE)
    
    # Show comparison
    if os.path.exists(OUTPUT_DIR):
        show_comp = input("\nShow before/after comparison? (yes/no): ")
        if show_comp.lower() in ['yes', 'y']:
            show_before_after_comparison(SOURCE_DIR, OUTPUT_DIR, num_samples=4)
    
    print("\n✓ Done! Ready to retrain with bias-free dataset.")

if __name__ == "__main__":
    main()