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
# IMPROVED Settings
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20  # Increased from 10
LEARNING_RATE = 0.001
DATA_DIR = "brain_tumor_dataset"

# Best model tracking
best_val_acc = 0.0
best_model_state = None

# -----------------------------
# Enhanced Transformations
# -----------------------------
# Use DIFFERENT transforms for training vs validation
# Training gets augmentation to learn from more variations
# Validation gets NO augmentation so we test on unaugmented images

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # DATA AUGMENTATION (only for training):
    transforms.RandomHorizontalFlip(p=0.5),      # 50% chance to flip
    transforms.RandomRotation(20),                # Rotate up to ±20 degrees
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),  # Random zoom
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary brightness/contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # NO augmentation for validation - just normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Load Data with Separate Transforms
# -----------------------------
def load_data(data_path):
    """Load images without transforms (we'll apply them later)"""
    categories = ['yes', 'no']
    data, labels = [], []

    for category in categories:
        class_num = categories.index(category)
        path = os.path.join(data_path, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img)
            labels.append(class_num)

    return data, labels

# Custom Dataset class to apply different transforms to train/val
class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load raw data
print("Loading dataset...")
images, labels = load_data(DATA_DIR)
print(f"Loaded {len(images)} images")

# Split indices
dataset_size = len(images)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

indices = list(range(dataset_size))
np.random.shuffle(indices)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create train/val datasets with different transforms
train_images = [images[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]
val_images = [images[i] for i in val_indices]
val_labels = [labels[i] for i in val_indices]

train_dataset = BrainTumorDataset(train_images, train_labels, train_transform)
val_dataset = BrainTumorDataset(val_images, val_labels, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}\n")

# -----------------------------
# IMPROVED Model Architecture
# -----------------------------
print("Building model...")
model = models.resnet18(weights="IMAGENET1K_V1")

# IMPROVEMENT 1: Unfreeze the last ResNet block for fine-tuning
# This allows the model to adapt the high-level features specifically for tumors
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 (the last convolutional block)
# By unfreezing it, we let it adapt these features specifically for brain tumors
for param in model.layer4.parameters():
    param.requires_grad = True

# IMPROVEMENT 2: Better classifier with BatchNorm
# - BatchNorm: Normalizes activations, makes training more stable
# - Larger hidden layer (256 instead of 128): More capacity to learn
# - Two dropout layers: Better regularization to prevent overfitting
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.BatchNorm1d(256),  # Normalize activations
    nn.ReLU(),
    nn.Dropout(0.5),  # Drop 50% during training
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

print("Model architecture created")
print(f"Unfroze layer4 for fine-tuning")
print(f"Enhanced classifier with BatchNorm\n")

# -----------------------------
# IMPROVEMENT 3: Loss and Optimizer with Weight Decay
# -----------------------------
criterion = nn.BCELoss()

# Optimize BOTH the unfrozen layer4 AND the classifier
# Using different learning rates 
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': LEARNING_RATE * 0.1},  # Lower LR for pretrained layers
    {'params': model.fc.parameters(), 'lr': LEARNING_RATE}  # Higher LR for new classifier
], weight_decay=0.0001)  # L2 regularization

# IMPROVEMENT 4: Learning Rate Scheduler
# Reduces learning rate when validation accuracy plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',           # Want to maximize accuracy
    factor=0.5,           # Reduce LR by half
    patience=3,           # Wait 3 epochs before reducing
)

print("Optimizer configured with different learning rates")
print("Learning rate scheduler enabled\n")

# -----------------------------
# IMPROVEMENT 5: Training with Early Stopping
# -----------------------------
train_losses = []
val_accuracies = []
val_losses = []
patience_counter = 0
early_stop_patience = 7  # Stop if no improvement for 7 epochs

print("Starting training...")
print("="*60)

for epoch in range(EPOCHS):
    # ----- TRAINING PHASE -----
    model.train()
    total_loss = 0
    for inputs, labels_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels_batch.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ----- VALIDATION PHASE -----
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    
    with torch.no_grad():
        for inputs, labels_batch in val_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels_batch.float())
            val_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)
    
    # Update learning rate based on validation accuracy
    scheduler.step(val_acc)
    
    # IMPROVEMENT 6: Save best model
    # Keep track of the best performing model
    # Even if later epochs overfit, still use the best one
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% NEW BEST!")
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # IMPROVEMENT 7: Early stopping
    # If validation accuracy doesn't improve for 7 epochs, stop training
    # Prevents wasting time and potential overfitting
    if patience_counter >= early_stop_patience:
        print(f"\n⚠ Early stopping triggered! No improvement for {early_stop_patience} epochs.")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        break

print("="*60)
print(f"\nTraining complete!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

# Load best model
model.load_state_dict(best_model_state)

# -----------------------------
# Enhanced Plotting
# -----------------------------
plt.figure(figsize=(15, 5))

# Plot 1: Training and Validation Loss
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Training Loss", color='blue')
plt.plot(val_losses, label="Validation Loss", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Validation Accuracy
plt.subplot(1, 3, 2)
plt.plot(val_accuracies, label="Validation Accuracy", color="green", marker='o')
plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best: {best_val_acc:.2f}%')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Over Time")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Overfitting Check
plt.subplot(1, 3, 3)
plt.plot(train_losses, label="Train Loss", color='blue')
plt.plot(val_losses, label="Val Loss", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfitting Check")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------
# Save Best Model
# -----------------------------
torch.save(model.state_dict(), "resnet18_brain_tumor_IMPROVED.pth")
print(f"\n✓ Best model saved as 'resnet18_brain_tumor_IMPROVED.pth'")
print(f"✓ This model achieved {best_val_acc:.2f}% validation accuracy")
