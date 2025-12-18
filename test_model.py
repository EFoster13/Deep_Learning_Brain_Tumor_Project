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
MODEL_PATH = "resnet18_brain_tumor_model.pth"
DATA_DIR = "brain_tumor_dataset"

# -----------------------------
# Load the Model
# -----------------------------
def load_model(model_path):
    """Load the trained ResNet18 model"""
    model = models.resnet18(weights=None)  # Don't load pretrained weights
    
    # Recreate the same classifier structure
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
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
# Transform (same as training)
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
    
    print("Testing model on all images...\n")
    
    for category in categories:
        class_num = categories.index(category)
        path = os.path.join(data_path, category)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img_rgb).unsqueeze(0)  # Add batch dimension
            
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
# Display Results
# -----------------------------
def display_results(labels, predictions, probabilities, images, filenames):
    """Display comprehensive test results"""
    
    # Calculate accuracy
    correct = sum([1 for l, p in zip(labels, predictions) if l == p])
    total = len(labels)
    accuracy = 100 * correct / total
    
    print("="*60)
    print(f"OVERALL TEST ACCURACY: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print("="*60)
    
    # Classification Report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print("-"*60)
    target_names = ['Tumor (yes)', 'No Tumor (no)']
    print(classification_report(labels, predictions, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Show some correct predictions
    print("\n" + "="*60)
    print("SAMPLE CORRECT PREDICTIONS:")
    print("="*60)
    correct_indices = [i for i in range(len(labels)) if labels[i] == predictions[i]]
    show_sample_predictions(correct_indices[:6], labels, predictions, 
                           probabilities, images, filenames, "CORRECT")
    
    # Show some incorrect predictions
    incorrect_indices = [i for i in range(len(labels)) if labels[i] != predictions[i]]
    if incorrect_indices:
        print("\n" + "="*60)
        print("SAMPLE INCORRECT PREDICTIONS:")
        print("="*60)
        show_sample_predictions(incorrect_indices[:6], labels, predictions, 
                               probabilities, images, filenames, "INCORRECT")

def show_sample_predictions(indices, labels, predictions, probabilities, images, filenames, title):
    """Display sample predictions in a grid"""
    if not indices:
        print("No images to display.")
        return
    
    num_images = min(len(indices), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    categories = ['TUMOR', 'NO TUMOR']
    
    for idx, i in enumerate(indices[:num_images]):
        axes[idx].imshow(images[i])
        
        true_label = categories[labels[i]]
        pred_label = categories[predictions[i]]
        confidence = probabilities[i] if predictions[i] == 1 else (1 - probabilities[i])
        
        color = 'green' if labels[i] == predictions[i] else 'red'
        axes[idx].set_title(f"True: {true_label}\nPred: {pred_label}\nConfidence: {confidence*100:.1f}%",
                           color=color, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(num_images, 6):
        axes[idx].axis('off')
    
    plt.suptitle(f"{title} PREDICTIONS", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Test Single Image
# -----------------------------
def test_single_image(model, image_path, transform):
    """Test model on a single image"""
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor).item()
        prediction = "TUMOR DETECTED" if output > 0.5 else "NO TUMOR"
        confidence = output if output > 0.5 else (1 - output)
    
    # Display result
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {prediction}\nConfidence: {confidence*100:.1f}%",
             fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Raw output: {output:.4f}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("Loading trained model...")
    model = load_model(MODEL_PATH)
    print("✓ Model loaded successfully!\n")
    
    # Test on all images
    labels, predictions, probabilities, images, filenames = test_all_images(
        model, DATA_DIR, transform
    )
    
    # Display comprehensive results
    display_results(labels, predictions, probabilities, images, filenames)
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
    
    # Optional: Test a specific image
    # Uncomment and modify the path below to test a single image:
    # test_single_image(model, "path/to/your/image.jpg", transform)