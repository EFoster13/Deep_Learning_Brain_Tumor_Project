import os
import matplotlib.pyplot as plt
import cv2

# Path to dataset
data_path = "brain_tumor_dataset"

# Show some images from each class
categories = ['yes', 'no']

# Loops through 'yes' and 'no' folders
for category in categories:
    # Takes first 5 images from each folder
    path = os.path.join(data_path, category)
    for img_name in os.listdir(path)[:5]: 
        img_path = os.path.join(path, img_name)
        # Reads image with OpenCV (which loads as BGR)
        img = cv2.imread(img_path)
        # Converts BGR to RGB (for correct colors in matplotlib)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Shows each image one at a time
        plt.imshow(img)
        plt.title(f"{category.upper()} - {img_name}")
        plt.axis('off')
        plt.show()