import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Load a few training images (already preprocessed)
from sklearn.model_selection import train_test_split
import os
import cv2

# ---- Load and preprocess just like your main script ----
IMG_SIZE = 128
data_path = "brain_tumor_dataset"
categories = ['yes', 'no']
data = []
labels = []

for category in categories:
    path = os.path.join(data_path, category)
    class_num = categories.index(category)
    for img_name in os.listdir(path)[:10]:  # Load only 10 images to test
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        data.append(img)
        labels.append(class_num)

# Convert to tensors
data = np.array(data)
labels = np.array(labels)
X_tensor = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)

# ---- Define augmentation transforms ----
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
])

# ---- Visualize original and augmented images ----
num_images = 5
for i in range(num_images):
    original = X_tensor[i]
    augmented = augmentation(original)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(original.permute(1, 2, 0).numpy())
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(augmented.permute(1, 2, 0).numpy())
    axs[1].set_title("Augmented")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()