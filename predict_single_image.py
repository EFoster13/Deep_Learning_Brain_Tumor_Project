import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "resnet18_brain_tumor_IMPROVED.pth"

# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path):
    """Load the trained model"""
    print("Loading model...")
    
    # Create model architecture (same as training)
    model = models.resnet18(weights=None)
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    print("✓ Model loaded successfully!")
    print("✓ Model accuracy: 98.81%\n")
    return model

# -----------------------------
# Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Predict Function
# -----------------------------
def predict_image(model, image_path):
    """
    Predict if an MRI scan contains a tumor
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the MRI image
    
    Returns:
        prediction: "TUMOR DETECTED" or "NO TUMOR"
        confidence: Confidence percentage (0-100)
        probability: Raw probability (0-1)
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor).item()
    
    # Interpret results
    # Output is probability of "no tumor" (class 1)
    # So we need to interpret it correctly
    if output > 0.5:
        prediction = "NO TUMOR"
        confidence = output * 100
    else:
        prediction = "TUMOR DETECTED"
        confidence = (1 - output) * 100
    
    return prediction, confidence, output, img_rgb

# -----------------------------
# Visualization Function
# -----------------------------
def display_prediction(image, prediction, confidence, probability, image_path):
    """Display the image with prediction results"""
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Original Image with Prediction
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    
    # Color code based on prediction
    color = 'red' if prediction == "TUMOR DETECTED" else 'green'
    
    plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.2f}%",
             fontsize=16, fontweight='bold', color=color, pad=20)
    plt.axis('off')
    
    # Plot 2: Confidence Meter
    plt.subplot(1, 2, 2)
    
    # Create confidence bar
    fig_ax = plt.gca()
    fig_ax.barh([0], [confidence], color=color, alpha=0.7, height=0.5)
    fig_ax.set_xlim(0, 100)
    fig_ax.set_ylim(-1, 1)
    fig_ax.set_xlabel('Confidence (%)', fontsize=12)
    fig_ax.set_title('Model Confidence', fontsize=14, fontweight='bold')
    fig_ax.set_yticks([])
    
    # Add confidence threshold line
    plt.axvline(x=50, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.legend()
    
    # Add text annotation
    plt.text(confidence/2, 0, f'{confidence:.1f}%', 
             ha='center', va='center', fontsize=20, fontweight='bold', color='white')
    
    plt.suptitle(f'Brain Tumor Detection Result\nImage: {os.path.basename(image_path)}',
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Raw Probability: {probability:.4f}")
    print("="*60)
    
    # Interpretation
    if confidence > 95:
        print("🎯 Very High Confidence - Model is very certain")
    elif confidence > 85:
        print("✅ High Confidence - Model is confident")
    elif confidence > 70:
        print("⚠️  Moderate Confidence - Consider additional review")
    else:
        print("❓ Low Confidence - Uncertain prediction")
    print("="*60 + "\n")

# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Main function to run predictions"""
    
    print("="*60)
    print(" "*15 + "BRAIN TUMOR PREDICTOR")
    print("="*60)
    print("Model Accuracy: 98.81%")
    print("Only 3 errors out of 253 test images!\n")
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Get image path from user
    print("="*60)
    print("Enter the path to your brain MRI image")
    print("Examples:")
    print("  - brain_tumor_dataset/yes/Y1.jpg")
    print("  - brain_tumor_dataset/no/N1.jpg")
    print("  - C:/Users/YourName/Desktop/brain_scan.jpg")
    print("="*60)
    
    image_path = input("\nImage path: ").strip()
    
    # Remove quotes if user included them
    image_path = image_path.strip('"').strip("'")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"\n❌ Error: File not found at {image_path}")
        print("Please check the path and try again.")
        return
    
    print(f"\n🔍 Analyzing image: {image_path}")
    print("Please wait...\n")
    
    try:
        # Make prediction
        prediction, confidence, probability, image = predict_image(model, image_path)
        
        # Display results
        display_prediction(image, prediction, confidence, probability, image_path)
        
        # Ask if user wants to analyze another image
        print("\nWould you like to analyze another image? (yes/no)")
        response = input().strip().lower()
        
        if response in ['yes', 'y']:
            print("\n" + "="*60 + "\n")
            main()  # Recursive call for another prediction
        else:
            print("\n✓ Thank you for using Brain Tumor Predictor!")
            print("="*60)
    
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        print("Please make sure the image is a valid MRI scan.")

# -----------------------------
# Run the predictor
# -----------------------------
if __name__ == "__main__":
    main()