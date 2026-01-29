# Causal AI: Breaking Spurious Correlations

> **An interactive demonstration of why standard deep learning fails on out-of-distribution data and how Invariant Risk Minimization (IRM) solves this through causal learning.**

## Overview

This project demonstrates the critical difference between **Standard Machine Learning** and **Causal Machine Learning** when dealing with spurious correlations. Through an interactive web application, you'll see firsthand how models can achieve 100% training accuracy yet completely fail on test data—and how causal learning prevents this.

**Key Insight**: Standard models learn shortcuts. When a digit is always red during training, the model learns "red = 0" instead of learning the digit's shape. This fails instantly when test data has random colors.

### Quick Stats

- **Training Accuracy**: Simple CNN & ResNet = 100%, IRM = 99.5%
- **Test Accuracy (OOD)**: Simple CNN & ResNet = ~10%, IRM = 97.6%
- **Interactive Demo**: Upload your own handwritten digits!

## The Problem

Standard deep learning models often learn "shortcuts" - spurious correlations that work well on training data but fail in real-world scenarios. This project uses a colored MNIST dataset where:

### Test Data:
- Digits are colored **randomly** (distribution shift!)
- This forces models to rely on **shape** rather than **color**

### Results:
| Model | Training Accuracy | Test Accuracy | Learned Feature |
|-------|------------------|---------------|-----------------|
| **Simple CNN** | ~100% | ~10% | Color (spurious) |
| **ResNet-18** | ~100% | ~10% | Color (spurious) |
| **IRM CNN** | ~98% | ~97% | Shape (causal) |

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (or download the files)
```bash
cd "Causal AI and Normal"
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

### First Time Usage

1. **Explore the sidebar** to see the color-to-digit mapping
2. **Select a model** (try Simple CNN first to see it fail!)
3. **Choose a test mode**:
   - Start with "Pre-generated Dataset" to see the problem
   - Then try "Upload Your Own Image" with a hand-drawn digit
4. **Compare models**: Switch to IRM CNN to see how causal learning solves the problem

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
   - Invariant Risk Minimization CNN will still predict 7 ✓ (learned the shape!)

This demonstrates that Simple CNN uses color as a shortcut, while IRM CNN learned the actual causal feature (digit shape).


## What is Invariant Risk Minimization (IRM)?

IRM is a causal learning framework that trains models to:

1. **Find invariant features**: Features that are predictive across all training environments
2. **Ignore spurious correlations**: Features that only work in specific contexts

### Key Difference

- **Simple CNN**: Minimizes standard cross-entropy loss → learns any pattern (including spurious correlations)
- **IRM CNN**: Minimizes `Risk + λ × InvariancePenalty` → forced to learn only invariant features across environments

## Key Concepts

### Spurious Correlation
A pattern in training data that doesn't reflect true causality. 

**Example in this project**: Color perfectly predicts digit in training data, but color doesn't *cause* the digit label.

**Classic example**: Ice cream sales correlate with drowning deaths, but ice cream doesn't cause drowning (both are caused by summer weather).

### Distribution Shift (Out-of-Distribution Testing)
When test data comes from a different distribution than training data. Standard ML models often fail dramatically under distribution shift.

**In this project**: 
- Training: Digit 0 is always Red, Digit 1 is always Green, etc.
- Testing: Colors are assigned randomly
- **Result**: Simple models fail because their learned "rule" (Color → Digit) no longer works

### Causal Features vs. Spurious Features

| Type | Example | Stability | Performance on OOD Data |
|------|---------|-----------|------------------------|
| **Spurious** | Color of digit | Changes across environments | Poor (~10%) |
| **Causal** | Shape of digit | Invariant across environments | Good (~97%) |

### Invariant Risk Minimization (IRM)
A training framework that:
1. Trains across multiple environments with different spurious correlations
2. Adds a penalty for features that aren't predictive across ALL environments
3. Forces the model to learn only invariant (causal) features

**Mathematical Formulation**:
```
minimize: Risk(Φ) + λ × Penalty(Φ)

where:
- Risk(Φ) = Average prediction error across environments
- Penalty(Φ) = Variance of optimal classifiers across environments
```

When the penalty is low, it means the same features work across all environments (invariant/causal).

## Technical Details

### Why Standard ML Fails

Standard Empirical Risk Minimization (ERM):
```
minimize: Average Loss across all training data
```

This allows the model to use **any** predictive feature, including spurious ones.

### How IRM Succeeds

```
minimize: Risk + λ × InvariancePenalty

InvariancePenalty = ||∇_w Risk_env1 - ∇_w Risk_env2||²
```

The penalty is high when the optimal classifier differs between environments. By minimizing it, IRM finds features that are predictive **everywhere**.

### Model Architectures

All three models use similar architectures:

**Simple CNN**:
- 2 Conv layers (3→32→64 channels)
- 2 FC layers (3136→128→10)
- ~270K parameters

**ResNet-18**:
- Modified for 28×28 images
- Smaller initial conv kernel (3×3 instead of 7×7)
- No initial maxpool
- ~11M parameters

**IRM CNN**:
- Same architecture as Simple CNN
- Different training objective (adds invariance penalty)
- ~270K parameters

**Key Insight**: IRM doesn't need more capacity—it needs better training!
