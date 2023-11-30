# ECG-Spectrogram-Classification
This project aims to classify ECG signals into normal and abnormal classes using machine learning techniques. The dataset used for training includes abnormal and normal ECG signals.

## Table of Contents
1. [Introduction](#introduction)
2. [Usage](#usage)
3. [STFT Visualizations](#stft-visualizations)
4. [Convolutional Neural Network](#convolutional-neural-network)
5. [How to Use the ECG Classifier](#how-to-use-the-ecg-classifier)

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
```python
import numpy as np
import matplotlib.pyplot as plt
import os

# Define STFT parameters
n_fft = 125
hop_length = 64

# Loop through all rows of "abNormalhHeartData" data and plot STFTs as PNG images
for i in range(len(abNormalhHeartData)):
    # Load the data
    data = np.array(abNormalhHeartData.iloc[i].values.flatten().tolist())

    # Compute FFT of the data
    fft_data = np.fft.fft(data)

    # Compute the STFT using sliding windows
    n_frames = 1 + (len(data) - n_fft) // hop_length
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
    for j in range(n_frames):
        frame = data[j * hop_length : j * hop_length + n_fft]
        fft_frame = np.fft.fft(frame)
        stft[:, j] = fft_frame[: n_fft // 2 + 1]

    # Convert complex values to magnitudes
    mag = np.abs(stft)

    # Visualize STFT as a heatmap and save as a PNG image
    plt.imshow(20 * np.log10(mag), cmap='magma', origin='lower', aspect='auto')
    plt.axis('off')
    plt.savefig(os.path.join('/content/drive/MyDrive/sujet_abnormal', f'figlog_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.clf()
# Convolutional Neural Network (CNN)
The CNN model used includes several layers of convolution and pooling, followed by fully connected layers. This model is trained on the images of spectrograms obtained by STFT.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(20, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(14, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
How to Use the ECG Classifier
Use the ECG_Classifier function to classify a new ECG signal. The spectrogram image is preprocessed, and the model's prediction is displayed with a graphical visualization.

# How to Use the ECG Classifier
Use the ECG_Classifier function to classify a new ECG signal. The spectrogram image is preprocessed, and the model's prediction is displayed with a graphical visualization.
```python
import cv2
def ECG_Classifier(img_path):
    img = cv2.imread(os.path.join(img_path))
    resize = tf.image.resize(img, (256,256))
    yhat = new_model.predict(np.expand_dims(resize/255, 0))
    yhat = float(yhat)*2
    if yhat > 0.5:
        ecg_class='Predicted class is Normal'
    else:
        ecg_class='Predicted class is Abnormal'
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB ))
    plt.title(ecg_class)
    plt.show()
