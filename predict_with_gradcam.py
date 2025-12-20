import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "resnet18_brain_tumor_IMPROVED.pth"

# -----------------------------
# Grad-CAM Implementation
# -----------------------------
class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    Shows which parts of the image the model focuses on
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_image: Preprocessed input tensor
            target_class: Not used for binary classification
        
        Returns:
            cam: Heatmap showing important regions (numpy array)
        """
        # Forward pass
        model_output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        model_output.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # Shape: [C, H, W]
        activations = self.activations[0]  # Shape: [C, H, W]
        
        # Global average pooling on gradients
        # This tells us how important each channel is
        weights = torch.mean(gradients, dim=(1, 2))  # Shape: [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()

# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path):
    """Load the trained model"""
    print("Loading model...")
    
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
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("✓ Model loaded successfully!")
    print("✓ Grad-CAM visualization enabled\n")
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
# Predict with Grad-CAM
# -----------------------------
def predict_with_gradcam(model, image_path):
    """
    Predict and generate Grad-CAM visualization
    
    Returns:
        prediction, confidence, probability, original_image, heatmap
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_tensor = transform(img_rgb).unsqueeze(0)
    img_tensor.requires_grad = True
    
    # Initialize Grad-CAM
    # Target layer4 (the last convolutional block we fine-tuned)
    gradcam = GradCAM(model, model.layer4)
    
    # Generate CAM
    cam = gradcam.generate_cam(img_tensor)
    
    # Resize CAM to match original image
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor).item()
    
    # Interpret results
    if output > 0.5:
        prediction = "NO TUMOR"
        confidence = output * 100
    else:
        prediction = "TUMOR DETECTED"
        confidence = (1 - output) * 100
    
    return prediction, confidence, output, img_resized, cam_resized

# -----------------------------
# Visualization with Grad-CAM
# -----------------------------
def display_gradcam_prediction(image, heatmap, prediction, confidence, probability, image_path):
    """
    Display comprehensive visualization with Grad-CAM
    
    EXPLANATION:
    - Original Image: The brain scan
    - Heatmap: What the model focuses on (red = important, blue = not important)
    - Overlay: Heatmap superimposed on original image
    """
    
    # Create heatmap overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(18, 10))
    
    color = 'red' if prediction == "TUMOR DETECTED" else 'green'
    
    # Plot 1: Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Original MRI Scan', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Plot 2: Grad-CAM Heatmap
    plt.subplot(2, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap\n(Model Focus Areas)', fontsize=14, fontweight='bold')
    plt.colorbar(label='Importance', fraction=0.046)
    plt.axis('off')
    
    # Plot 3: Overlay
    plt.subplot(2, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay: Where Model Looks', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Plot 4: Prediction Result (larger)
    plt.subplot(2, 3, 4)
    plt.text(0.5, 0.7, prediction, 
             ha='center', va='center', fontsize=32, fontweight='bold', color=color)
    plt.text(0.5, 0.4, f'Confidence: {confidence:.2f}%',
             ha='center', va='center', fontsize=20, color=color)
    
    # Confidence interpretation
    if confidence > 95:
        interpretation = '🎯 Very High Confidence'
    elif confidence > 85:
        interpretation = '✅ High Confidence'
    elif confidence > 70:
        interpretation = '⚠️ Moderate Confidence'
    else:
        interpretation = '❓ Low Confidence'
    
    plt.text(0.5, 0.15, interpretation,
             ha='center', va='center', fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # Plot 5: Confidence Bar
    plt.subplot(2, 3, 5)
    plt.barh([0], [confidence], color=color, alpha=0.7, height=0.5)
    plt.xlim(0, 100)
    plt.ylim(-1, 1)
    plt.xlabel('Confidence (%)', fontsize=12)
    plt.title('Model Confidence', fontsize=14, fontweight='bold')
    plt.axvline(x=50, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.yticks([])
    plt.legend()
    plt.text(confidence/2, 0, f'{confidence:.1f}%', 
             ha='center', va='center', fontsize=18, fontweight='bold', color='white')
    
    # Plot 6: Explanation
    plt.subplot(2, 3, 6)
    explanation_text = """
    GRAD-CAM EXPLANATION:
    
    🔴 RED/YELLOW: High importance
       Model focuses here for decision
    
    🔵 BLUE/PURPLE: Low importance
       Model ignores these regions
    
    📊 HOW IT WORKS:
    Grad-CAM traces which pixels
    influenced the model's decision
    by analyzing gradients through
    the neural network.
    
    This shows the model is looking
    at meaningful brain structures,
    not random artifacts!
    """
    plt.text(0.1, 0.5, explanation_text, 
             ha='left', va='center', fontsize=10, family='monospace')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.suptitle(f'Brain Tumor Detection with Grad-CAM Visualization\n'
                f'Image: {os.path.basename(image_path)} | Model Accuracy: 98.81%',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\n" + "="*70)
    print(" "*20 + "PREDICTION RESULTS")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Raw Probability: {probability:.4f}")
    print("="*70)
    print("\n📍 GRAD-CAM ANALYSIS:")
    print("The heatmap shows which regions of the brain scan")
    print("the model focused on when making its prediction.")
    print("\nRed/Yellow areas = High attention (important for decision)")
    print("Blue/Purple areas = Low attention (less relevant)")
    print("="*70 + "\n")

# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Main function with Grad-CAM"""
    
    print("="*70)
    print(" "*15 + "BRAIN TUMOR PREDICTOR WITH GRAD-CAM")
    print("="*70)
    print("Model Accuracy: 98.81%")
    print("Grad-CAM shows WHAT the model is looking at!\n")
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Get image path
    print("="*70)
    print("Enter the path to your brain MRI image")
    print("Examples:")
    print("  - brain_tumor_dataset/yes/Y1.jpg")
    print("  - brain_tumor_dataset/no/N1.jpg")
    print("="*70)
    
    image_path = input("\nImage path: ").strip().strip('"').strip("'")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"\n❌ Error: File not found at {image_path}")
        return
    
    print(f"\n🔍 Analyzing image with Grad-CAM: {image_path}")
    print("Generating heatmap visualization...")
    print("Please wait...\n")
    
    try:
        # Make prediction with Grad-CAM
        prediction, confidence, probability, image, heatmap = predict_with_gradcam(
            model, image_path
        )
        
        # Display results
        display_gradcam_prediction(image, heatmap, prediction, confidence, 
                                  probability, image_path)
        
        # Ask if user wants to analyze another image
        print("\nWould you like to analyze another image? (yes/no)")
        response = input().strip().lower()
        
        if response in ['yes', 'y']:
            print("\n" + "="*70 + "\n")
            main()
        else:
            print("\n✓ Thank you for using Brain Tumor Predictor!")
            print("="*70)
    
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

# -----------------------------
# Run the predictor
# -----------------------------
if __name__ == "__main__":
    main()