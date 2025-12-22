# Brain Tumor Classification using Deep Learning

A high-accuracy deep learning model for binary classification of brain MRI scans to detect the presence of tumors. This project uses transfer learning with ResNet18 and achieves **98.81% accuracy** on test data.

---

## Project Overview

This project implements a Convolutional Neural Network (CNN) for automated brain tumor detection from MRI scans. The model can classify brain scans into two categories:
- **Tumor Present** (yes)
- **No Tumor** (no)

The system leverages transfer learning with a pre-trained ResNet18 model, fine-tuned specifically for brain tumor detection.

---

## Performance Metrics & Critical Findings

### IMPORTANT DISCOVERY: Dataset Bias

Through Grad-CAM analysis and iterative validation, this project revealed **significant dataset bias** that inflated accuracy metrics. This section documents both the initial results and the bias discovery process.

### Initial Model Performance (Biased)

| Model | Accuracy | What It Actually Learned |
|-------|----------|--------------------------|
| **Initial Model** | 98.81% | Image size shortcuts (630×630 vs 225×225 pixels) ❌ |
| **After Center Crop** | 92.16% | Border artifacts (partial fix) ⚠️ |
| **After Circular Mask** | **96.08%** | Edge contrast patterns (bias persists) ⚠️ |

### Detailed Classification Report (Final Masked Model)

```
               precision    recall  f1-score   support
  Tumor (yes)       0.95      0.97      0.96       155
No Tumor (no)       0.96      0.93      0.95        98

     accuracy                           0.96       253
    macro avg       0.96      0.95      0.95       253
 weighted avg       0.96      0.96      0.96       253
```

**Key Findings:**
-  **Initial 98.81% accuracy was misleading** - model exploited dataset artifacts
-  **Grad-CAM revealed bias** - model focused on image size, borders, and edges instead of brain tissue
-  **Iterative preprocessing reduced bias** - from 98.81% → 92.16% → 96.08%
-  **Bias persisted despite aggressive preprocessing** - demonstrates dataset quality issues
-  **Final model more trustworthy** despite lower accuracy - focuses more on brain features

### Dataset Bias Analysis

**Discovered Biases:**
1. **Size Bias:** Tumor images averaged 630×630px, non-tumor 225×225px
2. **Border Bias:** Different edge brightness (14.3 intensity units difference on right border)
3. **Edge Contrast Bias:** Circular boundary characteristics differ between classes

**Preprocessing Attempts:**
1. Standard resizing → 98.81% (size bias remained)
2. Center cropping + standardization → 92.16% (border bias reduced)
3. Circular masking → 96.08% (edge bias persists)

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
├── brain_tumor_dataset/              # Original dataset (biased)
│   ├── yes/                          # MRI scans with tumors
│   └── no/                           # MRI scans without tumors
├── brain_tumor_dataset_fixed/        # Center-cropped dataset
│   ├── yes/                          
│   └── no/                           
├── brain_tumor_dataset_masked/       # Circular masked dataset (final)
│   ├── yes/                          
│   └── no/                           
├── brain_tumor_model2.py             # Original training script
├── brain_tumor_model_improved.py     # Enhanced training script (final)
├── test_model.py                     # Test original model
├── test_improved_model.py            # Test improved model
├── predict_single_image.py           # Simple prediction tool
├── predict_with_gradcam.py           # Grad-CAM visualization tool
├── dataset_diagnostic.py             # Bias analysis tool
├── fix_dataset_preprocessing.py      # Center crop preprocessing
├── apply_circular_mask.py            # Circular mask preprocessing
├── load_data.py                      # Data exploration script
├── test_augmentations.py             # Visualize data augmentation
├── .gitignore                        # Git ignore file
└── README.md                         # This file
```

**Note:** Model weight files (`.pth`) are not included in git due to size and the discovered bias issues.

---

##  Getting Started

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

## Explainable AI & Bias Discovery

### Grad-CAM Implementation

This project implements **Gradient-weighted Class Activation Mapping (Grad-CAM)** for model interpretability, which proved critical in discovering dataset bias.

**What Grad-CAM Revealed:**
- Initial model focused on **image corners and borders** rather than brain tissue
- Model exploited **size differences** (630×630 vs 225×225 pixels) as primary feature
- Even after preprocessing, model found **edge contrast patterns** to exploit

### Bias Discovery Process

#### 1. Initial Observation (Grad-CAM Analysis)
```
Tumor images: Model focused on corners ❌
No-tumor images: Model focused on brain tissue ✅
Conclusion: Model using shortcuts, not learning anatomy
```

#### 2. Dataset Diagnostic Analysis
Created diagnostic tools to quantify bias:
- **Corner brightness analysis:** Minimal difference (4.19 units)
- **Border analysis:** Significant difference (14.3 units on right border)
- **Dimension analysis:** Major size discrepancy (630×630 vs 225×225)
- **Average image comparison:** Clear structural differences

#### 3. Iterative Preprocessing Attempts

**Attempt 1: Center Cropping**
- Cropped to 85% of image
- Standardized to 400×400 pixels
- Result: 92.16% accuracy (bias reduced but present)

**Attempt 2: Circular Masking**
- Applied circular mask (90% diameter)
- Removed all corners completely
- Result: 96.08% accuracy (edge bias persists)

**Conclusion:** Dataset has inherent quality issues that preprocessing cannot fully resolve

### Visualization Examples

See `predict_with_gradcam.py` for interactive Grad-CAM visualization showing:
- Original MRI scan
- Grad-CAM heatmap (model attention)
- Overlay showing focus areas
- Prediction with confidence score

**Example Usage:**
```bash
python predict_with_gradcam.py
# Enter image path when prompted
# Observe where model focuses attention
```

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

## Key Learnings & Implications

### Critical Insights

1. **High Accuracy ≠ Good Model**
   - Achieved 98.81% by exploiting dataset artifacts
   - Lower accuracy (96.08%) with proper validation is more trustworthy
   - Metrics alone don't validate model quality

2. **Explainability is Essential**
   - Grad-CAM revealed hidden biases that accuracy metrics missed
   - Visual inspection of model attention is critical for medical AI
   - Black-box models are dangerous in high-stakes applications

3. **Dataset Quality > Model Complexity**
   - Sophisticated preprocessing couldn't fix fundamental data collection issues
   - Bias can persist through multiple remediation attempts
   - Data collection protocols matter more than model architecture

4. **Iterative Validation is Necessary**
   - Initial results looked great but were misleading
   - Multiple rounds of testing revealed progressive issues
   - Professional ML requires skepticism and validation

### Medical AI Considerations

**Why This Matters for Healthcare:**
- Deploying the 98.81% model would be dangerous despite high accuracy
- Medical AI must be interpretable and trustworthy
- Dataset bias can lead to systematic errors in clinical settings
- This project demonstrates proper ML validation methodology

### Lessons for Future Projects

**Always implement explainability** (Grad-CAM, SHAP, etc.)
**Visualize model attention** before deployment
**Question high accuracy** on small datasets
**Validate across multiple metrics** beyond accuracy
**Document limitations** honestly
**Prioritize dataset quality** in data collection phase

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

## Contributing

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

For questions or collaboration opportunities, please reach out via [ewfoster337@gmail.com](mailto:ewfoster337@gmail.com)

---

