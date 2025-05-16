import os
import matplotlib.pyplot as plt
import cv2

# Path to dataset
data_path = "brain_tumor_dataset"

# Show some images from each class
categories = ['yes', 'no']

for category in categories:
    path = os.path.join(data_path, category)
    for img_name in os.listdir(path)[:5]:  # Show 5 images per class
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"{category.upper()} - {img_name}")
        plt.axis('off')
        plt.show()