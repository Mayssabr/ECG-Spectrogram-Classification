# ECG-Spectrogram-Classification
This project aims to classify ECG signals into normal and abnormal classes using machine learning techniques. The dataset used for training includes abnormal and normal ECG signals.

# Table of Contents
Introduction
Dataset
Usage
STFT Visualizations
Convolutional Neural Network (CNN)
How to Use the ECG Classifier


# Introduction
This project aims to develop a model for classifying ECG signals to aid in the diagnosis of heart diseases. Recent advancements in artificial intelligence have demonstrated that deep neural networks, such as CNN and ResNet, can directly extract features from data and recognize cardiac arrhythmias.

# Dataset
The dataset used in this project comes from the PTB Diagnostic ECG Database, accessible on Kaggle and PhysioNet. It includes abnormal and normal ECG signals.

# Usage
To use the code in this project, follow these steps:
   Download the PTB Diagnostic ECG Database.
   Run the data exploration and create a database of spectrogram images obtained by STFT.
   Train the CNN model using the processed dataset.
   Use the ECG classifier to predict the class of new ECG signals.

# STFT Visualizations
The code includes visualizations of Short-Time Fourier Transforms (STFTs) of ECG signals using matplotlib.

# Convolutional Neural Network (CNN)
The CNN model used includes several layers of convolution and pooling, followed by fully connected layers. This model is trained on the images of spectrograms obtained by STFT.

# How to Use the ECG Classifier
Use the ECG_Classifier function to classify a new ECG signal. The spectrogram image is preprocessed, and the model's prediction is displayed with a graphical visualization.

