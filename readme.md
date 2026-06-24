# Attention-Based Modular CNN for CIFAR-10 Classification

## 📌 Overview
This repository contains a custom implementation of a deep Convolutional Neural Network (CNN) designed for high-accuracy image classification on the CIFAR-10 dataset. The project features a unique **modular architecture** where each intermediate block utilizes a self-attention mechanism to weight and fuse outputs from multiple independent convolutional paths.

By employing advanced training techniques such as **Label Smoothing**, **AdamW Optimization**, and **Cosine Annealing**, this model achieves a testing accuracy of **88.47%**.

## 🚀 Key Technical Features
- **Dynamic Feature Fusion:** Implemented an "Intermediate Block" structure where the output is a weighted sum of multiple convolutional layers. The weights are dynamically computed via a dedicated Fully Connected (FC) layer based on channel-wise averages.
- **Modular Design:** The network is built using 4 successive blocks with increasing channel depth (64 to 512), followed by a multi-layer perception (MLP) output head.
- **Advanced Training Pipeline:**
    - **Optimization:** AdamW with Weight Decay (0.01).
    - **Scheduling:** Cosine Annealing learning rate scheduler for smoother convergence.
    - **Regularization:** Integrated Batch Normalization, Dropout (0.3), and Label Smoothing (0.1) to prevent overfitting.
- **Custom Weight Initialization:** Utilized Xavier Initialization to stabilize the gradients during early training phases.

## 🛠 Tech Stack
- **Framework:** PyTorch
- **Vision:** Torchvision (Data Augmentation & Normalization)
- **Visualization:** Matplotlib
- **Diagnostics:** Torchsummary (Model Architecture Analysis)
- **Language:** Python 3.11

## 📊 Methodology & Results

### Architecture Summary
1. **Intermediate Blocks:** Each block receives an input $x$, processes it through $L$ independent $3\times3$ convolutions, and fuses them using a softmax-normalized attention vector $a$.
2. **Output Block:** Computes global average pooling followed by three Fully Connected layers to map features to the 10 CIFAR classes.

### Performance
The model was trained for 50 epochs. The validation loss consistently tracked the training loss, showing no signs of divergence or significant overfitting.

| Metric | Value |
| :--- | :--- |
| **Final Test Accuracy** | **88.47%** |
| **Best Epoch** | ~35 (Saturation point) |
| **Optimizer** | AdamW |
| **Loss Function** | Cross-Entropy with Label Smoothing |

> **Note:** Performance graphs (Loss per Batch and Accuracy per Epoch) can be found in the `notebooks/` directory or at the end of the project file.

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cifar10-modular-cnn.git
   ```
---

2. The `requirements.txt` file

```text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
matplotlib>=3.7.0
torchsummary>=1.5.1
tqdm>=4.65.0
```

---

This project was developed as part of the Neural Networks and Deep Learning (ECS7026P) module during my Master’s in Machine Learning at Queen Mary University of London. It represents an independent implementation of modular deep learning architectures.