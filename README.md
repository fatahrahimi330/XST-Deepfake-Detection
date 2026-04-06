# Explainable Spatiotemporal Deepfake Detection via Self-Supervised Integrated Fusion for Social Media Security

## Overview

Deepfake videos generated using advanced deep learning techniques pose a
serious threat to the integrity of social media platforms, digital
communication, and public trust. These manipulated videos can spread
misinformation, impersonate individuals, and influence public opinion.

This project presents an **Explainable Spatiotemporal Deepfake Detection
framework** that leverages **self-supervised learning and multimodal
fusion** to detect manipulated videos shared on social media platforms.

The proposed system analyzes both **spatial (frame-level)** and
**temporal (motion-level)** information from videos. A **self-supervised
feature learning strategy** allows the model to learn meaningful
representations without requiring large amounts of labeled data.
Furthermore, **explainability techniques** are integrated to provide
visual insights into the model's decisions, improving transparency and
trust.

The goal of this project is to develop a **robust, interpretable, and
scalable deepfake detection system suitable for social media security
applications.**

------------------------------------------------------------------------

## Key Features

-   Spatiotemporal deepfake detection using video frame and motion
    features
-   Self-supervised representation learning for improved feature
    extraction
-   Integrated multimodal feature fusion architecture
-   Explainable AI (XAI) for model interpretability
-   Designed for large-scale social media content analysis
-   Modular deep learning pipeline for experimentation

------------------------------------------------------------------------

## Project Objectives

-   Detect deepfake videos using spatial and temporal patterns
-   Reduce dependency on large labeled datasets using self-supervised
    learning
-   Improve detection robustness across different deepfake generation
    methods
-   Provide explainable visualizations of model decisions
-   Support security applications in social media platforms

------------------------------------------------------------------------

## Project Architecture

The system consists of the following modules:

1.  Data Preprocessing
    -   Video frame extraction
    -   Face detection and alignment
    -   Frame normalization
2.  Self-Supervised Learning
    -   Representation learning without manual labels
    -   Pretraining on large-scale video datasets
3.  Spatial Feature Extraction
    -   CNN models (ResNet / EfficientNet)
    -   Extract facial artifact features
4.  Temporal Modeling
    -   LSTM / Transformer / 3D CNN
    -   Capture motion inconsistencies
5.  Feature Fusion
    -   Combine spatial and temporal embeddings
6.  Deepfake Classification
    -   Binary classification: Real vs Fake
7.  Explainability
    -   Grad-CAM
    -   Attention visualization
    -   Saliency maps

------------------------------------------------------------------------

## Project Structure

    Explainable-ST-Deepfake-Detection/

    data/
        raw_videos/
        processed_frames/

    preprocessing/
    models/
    self_supervised/
    training/
    evaluation/
    explainability/
    utils/
    results/
    notebooks/

    requirements.txt
    README.md

------------------------------------------------------------------------

## Dataset

This project can work with several popular deepfake datasets:

-   FaceForensics++
-   DFDC (DeepFake Detection Challenge)
-   Celeb-DF
-   DeepFakeTIMIT

Preprocessing typically includes:

-   Frame extraction
-   Face detection
-   Cropping and resizing
-   Data augmentation

------------------------------------------------------------------------

## Installation

Clone the repository:

    git clone https://github.com/yourusername/explainable-st-deepfake-detection.git
    cd explainable-st-deepfake-detection

Create a virtual environment:

    python -m venv venv

Activate environment:

Mac/Linux

    source venv/bin/activate

Windows

    venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

## Requirements

Typical libraries:

-   Python 3.8+
-   PyTorch / TensorFlow
-   OpenCV
-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   tqdm

------------------------------------------------------------------------

## Training

Run training:

    python training/train.py

Training process:

1.  Load dataset
2.  Extract spatial features
3.  Learn temporal relationships
4.  Fuse features
5.  Train classifier
6.  Save trained model

------------------------------------------------------------------------

## Evaluation

Run evaluation:

    python evaluation/evaluate.py

Metrics:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   ROC-AUC

------------------------------------------------------------------------

## Explainability

Run Grad-CAM visualization:

    python explainability/gradcam.py

This module generates heatmaps showing which regions influenced the
model's decision.

------------------------------------------------------------------------

## Applications

-   Social media content moderation
-   Fake news detection
-   Digital media verification
-   Online identity protection
-   Video authenticity analysis

------------------------------------------------------------------------

## Future Work

-   Real-time deepfake detection
-   Multimodal detection (audio + video)
-   Transformer-based video architectures
-   Federated learning for privacy-preserving detection

------------------------------------------------------------------------

## License


------------------------------------------------------------------------

## Author

Abdul Fatah Rahimi\
Computer Science -- Data Science
