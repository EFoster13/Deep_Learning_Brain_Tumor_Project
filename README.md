# Brain Tumor Classification using Deep Learning

A high-accuracy deep learning model for binary classification of brain MRI scans to detect the presence of tumors. This project uses transfer learning with ResNet18 and achieves **98.81% accuracy** on test data.


---

## Project Overview

This project implements a Convolutional Neural Network (CNN) for automated brain tumor detection from MRI scans. The model can classify brain scans into two categories:
- **Tumor Present** (yes)
- **No Tumor** (no)

The system leverages transfer learning with a pre-trained ResNet18 model, fine-tuned specifically for brain tumor detection.

---

## Performance Metrics

### Model Comparison

| Model | Accuracy | Tumor Detection | No Tumor Detection | Correct Predictions |
|-------|----------|-----------------|-------------------|---------------------|
| **Original Model** | 89.72% | 97% recall | 79% recall | 227/253 |
| **Improved Model** | **98.81%** | **98.71%** | **98.98%** | **250/253** |
| **Improvement** | **+9.09%** | **+1.71%** | **+19.98%** | **+23 images** |

### Detailed Classification Report (Improved Model)

```
               precision    recall  f1-score   support
  Tumor (yes)       0.99      0.99      0.99       155
No Tumor (no)       0.98      0.99      0.98        98

     accuracy                           0.99       253
    macro avg       0.99      0.99      0.99       253
 weighted avg       0.99      0.99      0.99       253
```

**Key Achievements:**
- ✅ **98.81% overall accuracy** on 253 test images
- ✅ Only **3 misclassifications** out of 253 images
- ✅ **99% precision** - When predicting tumor, correct 99% of the time
- ✅ **99% recall** - Detects 99% of actual tumors
- ✅ **Balanced performance** across both classes

---

## Model Architecture

### Base Model: ResNet18 (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuned last convolutional block (layer4) for brain tumor detection
- Custom classifier head with BatchNorm

### Enhanced Classifier Architecture
```python
nn.Sequential(
    nn.Linear(512, 256),
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
```

### Key Technical Features
- **Transfer Learning**: Leverages pre-trained ResNet18 weights
- **Fine-Tuning**: Layer4 unfrozen for domain-specific adaptation
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout Regularization**: Prevents overfitting (0.5 and 0.4)
- **Data Augmentation**: Random flips, rotations, color jitter, and crops
- **Early Stopping**: Prevents overfitting (patience: 7 epochs)
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Differential Learning Rates**: 0.0001 for layer4, 0.001 for classifier

---

## Project Structure

```
AI_Brain_Tumor_Project/
├── brain_tumor_dataset/
│   ├── yes/              # MRI scans with tumors
│   └── no/               # MRI scans without tumors
├── brain_tumor_model2.py           # Original training script
├── brain_tumor_model_improved.py   # Improved training script (98.81%)
├── test_model.py                   # Test original model
├── test_improved_model.py          # Test improved model
├── load_data.py                    # Data exploration script
├── test_augmentations.py           # Visualize data augmentation
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI_Brain_Tumor_Project.git
   cd AI_Brain_Tumor_Project
   ```

2. **Create a virtual environment**
   ```bash
   # Using conda (recommended)
   conda create -n brain_tumor python=3.10
   conda activate brain_tumor
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision opencv-python matplotlib numpy scikit-learn seaborn
   ```

### Dataset Setup

1. Organize your brain MRI dataset in the following structure:
   ```
   brain_tumor_dataset/
   ├── yes/    # Images with tumors
   └── no/     # Images without tumors
   ```

2. **Verify dataset**
   ```bash
   python load_data.py
   ```
   This will display sample images from each class.

---

## Usage

### 1. Visualize Data Augmentation
```bash
python test_augmentations.py
```
Shows original vs augmented images to understand data preprocessing.

### 2. Train the Model

**Original Model (89.72% accuracy):**
```bash
python brain_tumor_model2.py
```

**Improved Model (98.81% accuracy):**
```bash
python brain_tumor_model_improved.py
```

Training will:
- Load and preprocess all images
- Apply data augmentation
- Train for up to 20 epochs (with early stopping)
- Save the best model automatically
- Display training curves

**Training Output:**
```
Epoch 1/20 | Train Loss: 0.5980 | Val Loss: 0.6030 | Val Acc: 86.27% ⭐ NEW BEST!
Epoch 2/20 | Train Loss: 0.4523 | Val Loss: 0.3756 | Val Acc: 92.16% ⭐ NEW BEST!
Epoch 3/20 | Train Loss: 0.3514 | Val Loss: 0.2587 | Val Acc: 94.12% ⭐ NEW BEST!
Epoch 4/20 | Train Loss: 0.2455 | Val Loss: 0.1964 | Val Acc: 98.04% ⭐ NEW BEST!
...
Best validation accuracy: 98.04%
```

### 3. Test the Model

**Test original model:**
```bash
python test_model.py
```

**Test improved model:**
```bash
python test_improved_model.py
```

Testing will provide:
- Overall accuracy
- Confusion matrix visualization
- Precision, recall, and F1-scores
- Confidence distribution plots
- Sample correct and incorrect predictions
- Per-class performance breakdown

---

## Technical Details

### Data Preprocessing
- **Image Size**: 224x224 pixels (ResNet18 input size)
- **Normalization**: ImageNet mean and std
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Color Space**: RGB

### Data Augmentation (Training Only)
- Random horizontal flip (50% probability)
- Random rotation (±20 degrees)
- Random resized crop (80-100% scale)
- Color jitter (brightness ±20%, contrast ±20%)

### Training Configuration
- **Optimizer**: Adam
  - Layer4 learning rate: 0.0001
  - Classifier learning rate: 0.001
  - Weight decay: 0.0001 (L2 regularization)
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Batch Size**: 32
- **Max Epochs**: 20
- **Early Stopping**: Patience of 7 epochs
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 3 epochs

### Model Improvements Over Baseline
1. **Fine-tuning layer4** instead of freezing all layers
2. **Larger classifier** (256 → 128 → 1) instead of (128 → 1)
3. **Batch Normalization** for training stability
4. **Enhanced data augmentation** with color jitter
5. **Differential learning rates** for layer4 vs classifier
6. **Early stopping** to prevent overfitting
7. **Best model tracking** to save optimal weights

---

## Results Visualization

The testing scripts generate several visualizations:

1. **Confusion Matrix**: Shows true positives, false positives, true negatives, false negatives
2. **Confidence Distribution**: Histogram of model prediction confidence for each class
3. **Sample Predictions**: Grid of images with predictions and confidence scores
4. **Training Curves**: Loss and accuracy over epochs

---

## Key Learnings

### Why Transfer Learning Works
- ResNet18 pre-trained on ImageNet already understands:
  - Edges and textures (early layers)
  - Shapes and patterns (middle layers)
  - Complex features (deep layers)
- We only need to teach it brain tumor-specific features

### Why Fine-Tuning Helps
- Freezing all layers: Model can't adapt features
- Training from scratch: Needs massive dataset
- **Fine-tuning last block**: Best of both worlds! ✅

### Why BatchNorm Matters
- Normalizes layer inputs during training
- Allows higher learning rates
- Acts as regularization
- Makes training more stable

### Medical AI Considerations
- **High recall is critical**: Missing a tumor (false negative) is worse than a false alarm
- **Balanced performance**: Both classes need high accuracy
- **Interpretability**: Important for medical applications (future work: attention maps)

---

## Future Improvements

- [ ] Multi-class classification (tumor types: glioma, meningioma, pituitary)
- [ ] Grad-CAM visualization to show which regions influence predictions
- [ ] Ensemble models (combine multiple architectures)
- [ ] Test on external datasets for generalization
- [ ] Web application for easy deployment
- [ ] Integration with DICOM medical imaging standards
- [ ] Uncertainty quantification for predictions

---

## References

- **ResNet Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Transfer Learning**: [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526)
- **Medical Image Analysis**: [Deep Learning in Medical Image Analysis](https://www.sciencedirect.com/science/article/pii/S1361841517301135)

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Author

Your Name
- GitHub: [@EFoster13](https://github.com/EFoster13)
- LinkedIn: [Ethan Foster](https://linkedin.com/in/ethan-foster-3916b7329)

---

## Acknowledgments

- This project uses a publicly available Brain Tumor MRI dataset from Kaggle, consisting of labeled MRI images categorized by tumor presence. The dataset is used to train and evaluate deep learning models for automated brain tumor detection.
- PyTorch and torchvision teams
- ResNet architecture by Microsoft Research
- Open source community

---

## Contact

For questions or collaboration opportunities, please reach out via [ewfoster337@gmail.com]

---
