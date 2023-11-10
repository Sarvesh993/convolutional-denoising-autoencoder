# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement 

A Denoising Autoencoder is a modification on the autoencoder to prevent the network learning the identity function.This helps to avoid the autoencoders to copy the input to the output without learning features about the data. These autoencoders take a partially corrupted input while training to recover the original undistorted input. The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.


## DESIGN STEPS

### STEP 1:

Import the necessary libraries and download the mnist dataset.

### STEP 2:
Load the dataset and scale the values for easier computation.

### STEP 3:
Add noise to the images randomly for the process of denoising it with the convolutional denoising autoencoders for both the training and testing sets.

### STEP 4:
Build the Neural Model for convolutional denoising autoencoders using Convolutional, Pooling and Up Sampling layers. Make sure the input shape and output shape of the model are identical.

### STEP 5:
Pass test data for validating manually. Compile and fit the created model.

### STEP 6:
Plot the Original, Noisy and Reconstructed Image predictions for visualization.

### STEP 7:
End the program.

Write your own steps

## PROGRAM

DEVELOPE BY: Sarveshvaran P

REG NO: 212221230090
```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.8
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3, 3),activation = 'relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation = 'relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=3,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
                
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original:
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy:
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction:
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/Sarvesh993/convolutional-denoising-autoencoder/assets/94881923/85956fa8-7e65-4bc9-8cd6-40725cde2eb4)

### Model Summary

![image](https://github.com/Sarvesh993/convolutional-denoising-autoencoder/assets/94881923/931aecf2-e433-4e55-a7fc-6385b3e5c0eb)

### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/Sarvesh993/convolutional-denoising-autoencoder/assets/94881923/34bb250d-faaf-4976-a9d7-96390efcd699)




## RESULT

Thus we have successfully developed a convolutional autoencoder for image denoising application.
