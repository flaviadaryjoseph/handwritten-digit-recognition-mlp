# ✍️ Handwritten Digit Recognition Using MLP

## Project Overview

This project implements a Multi-Layer Perceptron (MLP) Artificial Neural Network
for recognizing handwritten digits from 0 to 9 using the MNIST dataset.

## Dataset

The MNIST dataset contains:

- 60,000 training images
- 10,000 testing images
- 28 × 28 grayscale images
- 10 classes (digits 0–9)

## Data Preprocessing

The images are:

1. Converted to floating-point values
2. Normalized from 0–255 to 0–1
3. Flattened from 28 × 28 into 784 features

## Model Architecture

```text
Input Layer: 784 neurons
        ↓
Dense Layer: 256 neurons + ReLU
        ↓
Dropout: 20%
        ↓
Dense Layer: 128 neurons + ReLU
        ↓
Output Layer: 10 neurons + Softmax