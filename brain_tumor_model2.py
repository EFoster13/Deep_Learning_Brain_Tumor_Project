import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import models, transforms
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 224           # ResNet expects 224x224 images
BATCH_SIZE = 32          # Process 32 images at a time
EPOCHS = 10              # Train for 10 complete passes through data
LEARNING_RATE = 0.001
DATA_DIR = "brain_tumor_dataset"

# -----------------------------
# Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),                           # Convert to PIL format
    transforms.Resize((IMG_SIZE, IMG_SIZE)),           # Resize to 224x224
    transforms.RandomHorizontalFlip(),                 # Augmentation: flip
    transforms.RandomRotation(15),                     # Augmentation: rotate ±15°
    transforms.ToTensor(),                             # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Normalize using ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Load Data with Transform
# -----------------------------
def load_data_with_transforms(data_path, transform):
    # Assign label: 0 for 'yes' (tumor), 1 for 'no' (no tumor)
    categories = ['yes', 'no']
    data, labels = [], []

    for category in categories:
        class_num = categories.index(category)
        path = os.path.join(data_path, category)
        # Load every image in the folder
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)  # Apply all transformations
            data.append(img)
            labels.append(class_num)

    return torch.stack(data), torch.tensor(labels)

X, y = load_data_with_transforms(DATA_DIR, transform)

# -----------------------------
# Data Splitting
# -----------------------------
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))          # 80% for training
val_size = len(dataset) - train_size          # 20% for validation
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# -----------------------------
# Load Pretrained ResNet18
# -----------------------------
model = models.resnet18(weights="IMAGENET1K_V1")         # Load pretrained weights

# Freeze all convolutional layers (don't retrain them)
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier layer
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),      # 512 inputs → 128 neurons
    nn.ReLU(),                         # Activation function
    nn.Dropout(0.4),                   # Drop 40% of neurons (prevents overfitting)
    nn.Linear(128, 1),                 # 128 → 1 output
    nn.Sigmoid()                       # Squash output to 0-1 (probability)

)

# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.BCELoss()              # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Training Loop
# -----------------------------
train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()             # Set to training mode
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()                       # Reset gradients
        outputs = model(inputs).squeeze()           # Make predictions
        loss = criterion(outputs, labels.float())   # Calculate loss
        loss.backward()                             # Calculate gradients
        optimizer.step()                            # Update weights
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation
    model.eval()          # Set to evaluation mode (no dropout, etc.)
    correct = 0
    total = 0
    with torch.no_grad():          # Don't calculate gradients (saves memory)
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            preds = (outputs > 0.5).float()          # >0.5 = tumor, <0.5 = no tumor
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    val_accuracies.append(acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

# -----------------------------
# Plotting Results
# -----------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")      # Plot training loss over time
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy", color="green")    # Plot validation accuracy over time
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "resnet18_brain_tumor_model.pth") # Save the trained model weights
