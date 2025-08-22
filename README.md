# Human Activity Recognition with Federated Learning and KAN

This project implements a federated learning system using a Kolmogorov-Arnold Network (KAN) model for human activity recognition using the WISDM dataset.

## Overview

The system processes accelerometer data from smartphones to classify human activities using a novel KAN-based neural network architecture. The implementation features federated learning, allowing model training across multiple distributed clients while keeping data localized.

## Dataset

The project uses the WISDM (Wireless Sensor Data Mining) dataset containing accelerometer data from smartphones. The dataset includes six activity classes:
- Walking
- Jogging
- Upstairs
- Downstairs
- Sitting
- Standing  

Dataset link - https://www.cis.fordham.edu/wisdm/dataset.php

## Model Architecture

The KANModel consists of:
- Temporal KAN layer for feature extraction
- Bidirectional LSTM for sequence modeling
- Attention mechanism for focusing on important time steps
- Classifier for final activity prediction

## Federated Learning Setup

The implementation includes:
- Federated client management
- Model training on distributed clients
- Federated averaging for model aggregation
- Client selection with configurable fraction participation

## Key Features

1. **Data Preprocessing**: 
   - Sequence creation with sliding windows
   - Per-user normalization using StandardScaler
   - Label encoding for activity classes

2. **Federated Learning**:
   - Client-server architecture
   - Configurable training rounds and client participation
   - Secure model aggregation

3. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix visualization
   - ROC curve analysis
   - Training/validation performance tracking

## Installation

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm
