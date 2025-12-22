import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "resnet18_brain_tumor_IMPROVED.pth"  # Using improved model
DATA_DIR = "brain_tumor_dataset"

# -----------------------------
# Load the IMPROVED Model
# -----------------------------
def load_improved_model(model_path):
    """Load the improved ResNet18 model with fine-tuned layer4"""
    model = models.resnet18(weights=None)
    
    # Recreate the IMPROVED classifier structure (matches training)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),  # Larger: 256 instead of 128
        nn.BatchNorm1d(256),            # BatchNorm added
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),            # BatchNorm added
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# -----------------------------
# Transform (same as validation - NO augmentation)
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Test on All Images
# -----------------------------
def test_all_images(model, data_path, transform):
    """Test model on all images and return results"""
    categories = ['yes', 'no']
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_images = []
    all_filenames = []
    
    print("Testing improved model on all images...\n")
    
    for category in categories:
        class_num = categories.index(category)
        path = os.path.join(data_path, category)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_rgb).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                output = model(img_tensor).item()
                prediction = 1 if output > 0.5 else 0
            
            all_labels.append(class_num)
            all_predictions.append(prediction)
            all_probabilities.append(output)
            all_images.append(img_rgb)
            all_filenames.append(f"{category}/{img_name}")
    
    return all_labels, all_predictions, all_probabilities, all_images, all_filenames

# -----------------------------
# Display Comparison Results
# -----------------------------
def display_comparison_results(labels, predictions, probabilities, images, filenames):
    """Display comprehensive test results with comparison to original"""
    
    # Calculate metrics
    correct = sum([1 for l, p in zip(labels, predictions) if l == p])
    total = len(labels)
    accuracy = 100 * correct / total
    
    print("="*70)
    print(" "*20 + "IMPROVED MODEL RESULTS")
    print("="*70)
    print(f"OVERALL TEST ACCURACY: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print("="*70)
    
    # Show improvement comparison
    original_acc = 89.72  # From your previous test
    improvement = accuracy - original_acc
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"   Original Model:  {original_acc:.2f}%")
    print(f"   Improved Model:  {accuracy:.2f}%")
    print(f"   Improvement:     +{improvement:.2f}% 🚀")
    print("="*70)
    
    # Detailed classification report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print("-"*70)
    target_names = ['Tumor (yes)', 'No Tumor (no)']
    print(classification_report(labels, predictions, target_names=target_names))
    
    # Confusion Matrix with enhanced styling
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 16})
    plt.title('Confusion Matrix - Improved Model\n98.04% Validation Accuracy', 
              fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy text
    plt.text(1, -0.3, f'Overall Accuracy: {accuracy:.2f}%', 
             ha='center', fontsize=14, fontweight='bold',
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate per-class metrics
    tumor_correct = sum([1 for i, (l, p) in enumerate(zip(labels, predictions)) 
                        if l == 0 and p == 0])
    tumor_total = sum([1 for l in labels if l == 0])
    no_tumor_correct = sum([1 for i, (l, p) in enumerate(zip(labels, predictions)) 
                           if l == 1 and p == 1])
    no_tumor_total = sum([1 for l in labels if l == 1])
    
    print("\n" + "="*70)
    print("PER-CLASS BREAKDOWN:")
    print("-"*70)
    print(f"Tumor Detection (yes):    {tumor_correct}/{tumor_total} correct "
          f"({100*tumor_correct/tumor_total:.2f}%)")
    print(f"No Tumor Detection (no):  {no_tumor_correct}/{no_tumor_total} correct "
          f"({100*no_tumor_correct/no_tumor_total:.2f}%)")
    print("="*70)
    
    # Show confidence distribution
    tumor_probs = [probabilities[i] for i in range(len(labels)) if labels[i] == 0]
    no_tumor_probs = [probabilities[i] for i in range(len(labels)) if labels[i] == 1]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(tumor_probs, bins=20, color='red', alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Predicted Probability (Tumor)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Confidence Distribution - Tumor Images', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(no_tumor_probs, bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Predicted Probability (Tumor)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Confidence Distribution - No Tumor Images', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show sample correct predictions
    print("\n" + "="*70)
    print("SAMPLE CORRECT PREDICTIONS:")
    print("="*70)
    correct_indices = [i for i in range(len(labels)) if labels[i] == predictions[i]]
    show_sample_predictions(correct_indices[:6], labels, predictions, 
                           probabilities, images, filenames, "CORRECT")
    
    # Show incorrect predictions if any
    incorrect_indices = [i for i in range(len(labels)) if labels[i] != predictions[i]]
    if incorrect_indices:
        print("\n" + "="*70)
        print("INCORRECT PREDICTIONS (for analysis):")
        print("="*70)
        show_sample_predictions(incorrect_indices, labels, predictions, 
                               probabilities, images, filenames, "INCORRECT")
    else:
        print("\n" + "="*70)
        print("PERFECT SCORE! No incorrect predictions!")
        print("="*70)

def show_sample_predictions(indices, labels, predictions, probabilities, images, filenames, title):
    """Display sample predictions in a grid"""
    if not indices:
        print("No images to display.")
        return
    
    num_images = min(len(indices), 6)
    rows = (num_images + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    categories = ['TUMOR', 'NO TUMOR']
    
    for idx, i in enumerate(indices[:num_images]):
        axes[idx].imshow(images[i])
        
        true_label = categories[labels[i]]
        pred_label = categories[predictions[i]]
        confidence = probabilities[i] if predictions[i] == 1 else (1 - probabilities[i])
        
        color = 'green' if labels[i] == predictions[i] else 'red'
        axes[idx].set_title(f"True: {true_label}\nPred: {pred_label}\n"
                           f"Confidence: {confidence*100:.1f}%",
                           color=color, fontweight='bold', fontsize=11)
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"{title} PREDICTIONS - Improved Model", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("="*70)
    print(" "*15 + "TESTING IMPROVED MODEL")
    print("="*70)
    print("\nLoading improved model...")
    model = load_improved_model(MODEL_PATH)
    print("✓ Improved model loaded successfully!")
    print("✓ Model achieved 98.04% validation accuracy during training\n")
    
    # Test on all images
    labels, predictions, probabilities, images, filenames = test_all_images(
        model, DATA_DIR, transform
    )
    
    # Display comprehensive results
    display_comparison_results(labels, predictions, probabilities, images, filenames)
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)
