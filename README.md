# Causal AI: Breaking Spurious Correlations

> **An interactive demonstration of how standard CNNs learn shortcuts on colored MNIST, and an experiment testing when Invariant Risk Minimization (IRM) actually helps.**

## Overview

When a digit is always red during training, a CNN learns "red = 0" instead of the digit's shape, and fails as soon as test colors are random. This project shows that failure with a Simple CNN and ResNet-18, then compares standard training (ERM) with IRM on the same training data.

The main finding: IRM only beats standard training when the spurious feature (color) is more predictive than the causal one (shape) during training. When shape is already the better clue, a plain CNN learns it on its own and IRM adds nothing.

## Results

All test sets are built from the MNIST **test split**, so no test image is seen during training.

### 1. Shortcut learning

Simple CNN and ResNet-18 trained on data where every digit always has its label's color, tested on random colors:

| Model | Training Accuracy | Test Accuracy (random colors) |
|-------|------------------|-------------------------------|
| **Simple CNN** | 100% | 9.63% |
| **ResNet-18** | 99.96% | 10.31% |

Both models learn color, not shape, and drop to chance level.

### 2. ERM vs IRM on the same environments

Two training environments from disjoint halves of the MNIST train split. Color matches the label 90% of the time in env 1 and 80% in env 2. The ERM baseline uses the same model, data and schedule as IRM, only without the penalty. IRM switches the penalty on after 6 warm-up epochs.

**Without label noise**

| Model | Env 1 (train) | Test, 10% color corr. | Test, random colors |
|-------|---------------|-----------------------|---------------------|
| **ERM** | 99.94% | 97.58% | 97.43% |
| **IRM** | 99.46% | 96.36% | 95.89% |

IRM gives no benefit here. Shape always predicts the label while color is wrong 10-20% of the time, so plain ERM already learns shape.

**With 25% label noise** (the setting used in the IRM paper)

Each label is swapped for a different random digit with probability 0.25, before coloring. Shape now predicts the label only 75% of the time, less often than color (80-90%), so color becomes the easier shortcut. The best achievable test accuracy is 75%.

| Model | Env 1 (train) | Test, 10% color corr. | Test, random colors |
|-------|---------------|-----------------------|---------------------|
| **ERM** | 91.40% | 26.83% | 18.86% |
| **IRM** | 78.19% | 43.64% | 39.88% |

ERM latches onto color and collapses on both test sets. IRM gives up training accuracy and roughly doubles test accuracy, though it stays well below the 75% ceiling. Results are from a single seed.

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Paribartan-Timalsina/Causal-vs.-Normal-CNN.git
cd Causal-vs.-Normal-CNN
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0.0` - PyTorch for deep learning
- `torchvision>=0.15.0` - For datasets and model architectures
- `streamlit>=1.28.0` - Web application framework
- `matplotlib>=3.7.0` - Plotting and visualization
- `numpy>=1.24.0` - Numerical operations
- `Pillow>=9.5.0` - Image processing

### Running the Web App

```bash
streamlit run app.py
```

The app will automatically:
- Open in your default browser (usually at `http://localhost:8501`)
- Download MNIST data if not already present (~11MB)
- Load pre-trained model weights from the `models/` folder

### Training the Models

Run the training scripts from the repository root. Each one saves its weights to `models/`, where the app loads them from.

```bash
python -m training.train_simple_cnn
python -m training.train_resnet
python -m training.train_irm --compare-erm                     # IRM vs ERM, no label noise
python -m training.train_irm --compare-erm --label-noise 0.25  # IRM paper setting
```

All scripts take `--epochs`, `--lr`, `--batch-size`, `--seed`, `--data-dir` and `--out`. `train_irm.py` also takes `--warmup`, `--penalty-weight` and `--label-noise`; `--compare-erm` trains an ERM baseline on the same environments and prints both models' accuracies. A GPU is recommended, ResNet-18 in particular is slow on CPU.

The same experiments are also in `CNN and IRM.ipynb` (made for Google Colab). Set `LABEL_NOISE` in the IRM data cell to `0.0` or `0.25` to reproduce the two IRM tables.

### First Time Usage

1. **Explore the sidebar** to see the color-to-digit mapping
2. **Select a model** (try Simple CNN first to see it fail!)
3. **Choose a test mode**:
   - Start with "Pre-generated Dataset" to see the problem
   - Then try "Upload Your Own Image" with a hand-drawn digit
4. **Compare models**: Switch to IRM CNN and see it rely on shape

## Demo Scenario

### See the Problem in Action

1. **Select "Simple CNN"** and use pre-generated dataset
2. **Observe**: Model gets ~10% accuracy (random guessing!)
3. **Notice**: When it makes errors, the app shows "Fooled by [Color] color!"
4. **Why?**: The model learned "If Red → predict 0" which fails when 0 is now Blue

### Test with Your Own Image

1. Draw a **"7"** on paper and take a photo
2. **Upload** the image
3. **Apply Purple color** (which represents 7 in training)
   - Simple CNN will predict 7 ✓ (but for the wrong reason!)
4. **Change to Red color** (which represents 0)
   - Simple CNN will predict 0 ✗ (fooled by color!)
   - IRM CNN will still predict 7 ✓ (it relies on shape)

The IRM model in the app is trained without label noise. Its advantage over the Simple CNN comes mostly from its training environments, where color is not always reliable, rather than from the IRM penalty (see Results).

## What is Invariant Risk Minimization (IRM)?

IRM is a causal learning framework that trains models to:

1. **Find invariant features**: Features that are predictive across all training environments
2. **Ignore spurious correlations**: Features whose relationship with the label changes between environments

### Key Difference

- **ERM (standard training)**: Minimizes cross-entropy loss over all training data → uses whatever feature predicts best, including spurious ones
- **IRM**: Minimizes `Risk + λ × Penalty` → prefers features for which the same classifier is optimal in every environment

## Key Concepts

### Spurious Correlation
A pattern in training data that doesn't reflect true causality.

**Example in this project**: Color predicts the digit in training data, but color doesn't *cause* the digit label.

**Classic example**: Ice cream sales correlate with drowning deaths, but ice cream doesn't cause drowning (both are caused by summer weather).

### Distribution Shift (Out-of-Distribution Testing)
When test data comes from a different distribution than training data. Standard ML models often fail dramatically under distribution shift.

**In this project**:
- Training: Digit 0 is always (or mostly) Red, Digit 1 is Green, etc.
- Testing: Colors are assigned randomly
- **Result**: Models that learned "Color → Digit" fail because that rule no longer holds

### Causal Features vs. Spurious Features

| Type | Example | Stability across environments |
|------|---------|-------------------------------|
| **Spurious** | Color of digit | Changes (90% in env 1, 80% in env 2) |
| **Causal** | Shape of digit | Stays the same in every environment |

### Label Noise
Labels swapped for a wrong digit on purpose. It makes shape an imperfect clue (75%) while color stays a stronger one (80-90%), so a standard model is tempted to use color. This is the situation IRM is designed for, and the same setup is used in the original IRM paper.

## Technical Details

### Why Standard ML Fails

Standard Empirical Risk Minimization (ERM):
```
minimize: Average loss across all training data
```

This lets the model use **any** predictive feature. If color predicts the label better than shape in the training data, the model uses color.

### The IRM Objective (IRMv1)

```
minimize over Φ:   Σ_e R_e(Φ)  +  λ · Σ_e ‖ ∇_w R_e(w · Φ) |_{w=1} ‖²
```

- `R_e` is the cross-entropy loss in environment `e`
- `Φ` is the network, and `w` is a dummy scale on its output, fixed at 1.0
- The penalty is the squared gradient of each environment's loss with respect to `w`

The penalty is zero only when scaling the classifier would not reduce the loss in any environment, meaning the same classifier is optimal everywhere. Features like color, whose link to the label changes between environments, cannot satisfy this, so the penalty pushes the model away from them.

Training details: λ = 1000, switched on after 6 warm-up epochs of plain ERM. When the penalty is on, the loss is divided by λ and the Adam optimizer is reset to keep training stable, following the original IRM implementation.

## Reference

Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). *Invariant Risk Minimization.* arXiv:1907.02893.
