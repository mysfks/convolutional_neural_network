# CIFAR-10 Image Classification using TensorFlow

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). CIFAR-10 consists of 60,000 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. The classes are mutually exclusive and do not overlap. In this README, we will describe the dataset, the model architecture, the steps for training the model, and how to evaluate it.

## Table of Contents
- [Dataset Details](#dataset-details)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation and Prediction](#evaluation-and-prediction)
- [How to Run the Code](#how-to-run-the-code)
- [Dependencies](#dependencies)
- [Further Improvements](#further-improvements)

---

## Dataset Details

The CIFAR-10 dataset, loaded from Keras, is divided into:

- **Training Set:** 50,000 images
- **Testing Set:** 10,000 images

Each image in the dataset is a 32x32 color image that belongs to one of the following 10 classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is preloaded by Keras when you use the `tf.keras.datasets.cifar10.load_data()` function.

### Structure:
- The images (`X_train`, `X_test`) have dimensions `(32x32x3)` representing width, height, and 3 color channels (RGB). 
- The labels (`y_train`, `y_test`) are integer-encoded class values ranging from 0 to 9.

---

## Model Architecture

The CNN model used in this project is constructed using TensorFlow and Keras. The model consists of the following layers:

1. **Input Layer (Conv2D + MaxPooling2D)**  
   The first convolutional layer applies 32 filters of size (3x3) and uses the ReLU activation function. It is followed by a max pooling layer that down-samples the feature maps spatially by a factor of 2.
   
2. **Second Layer (Conv2D + MaxPooling2D)**  
   This layer has 64 filters with ReLU activation and is followed by another max pooling layer.

3. **Third Layer (Conv2D)**  
   Another convolutional layer with 64 filters is used to continue learning deeper features from the data.

4. **Flatten Layer**  
   This layer flattens the 3D data into a 1D vector to transition from convolutional layers to fully connected layers.

5. **First Dense Layer + Dropout**  
   A fully connected (dense) layer with 64 neurons and ReLU activation prepares the model for classification. A dropout layer follows it to reduce overfitting by randomly setting 50% of the node outputs to zero during training.

6. **Second Dense Layer + Dropout**  
   Another fully connected layer similar to the previous layer is added with 64 neurons, followed by another dropout layer.

7. **Output Layer**  
   The final output layer has 10 neurons (corresponding to the 10 classes) and uses the softmax activation function to output the class probabilities.

The model uses the Adam optimizer and `sparse_categorical_crossentropy` loss function, which is suitable for multi-class classification.

### Model Summary:
- Conv2D (Input: 32x32x3)
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Conv2D
- Flatten
- Dense (64 neurons)
- Dropout (0.5)
- Dense (64 neurons)
- Dropout (0.5)
- Dense (10 neurons, Softmax)

---

## Training Process

### Steps:
1. **Data Preprocessing**:
   - Split the original training set into a training set (first 40,000 images) and a validation set (last 10,000 images).
   - Normalize the image pixel values to a range of `[0, 1]` by dividing them by `255`.
   
2. **Model Compilation**:
   The model is compiled using the Adam optimizer and the `sparse_categorical_crossentropy` loss.

3. **Model Training**:
   The model is trained for **50 epochs** with a batch size of **128**. During training:
   - Training loss and accuracy are monitored.
   - Validation loss and accuracy are calculated at the end of each epoch to gauge the model's ability to generalize.

4. **Loss and Accuracy Visualization**:
   After training, the loss and accuracy vs. epochs are plotted for both the training and validation sets.

---

## Evaluation and Prediction

After training, the model was evaluated on the test set using the `model.evaluate()` function, achieving a final test accuracy.

A prediction can be made for any specific test sample. The pixel values of the test image are reshaped to fit the expected input dimensions of the model — `(1, 32, 32, 3)`, and the model's `predict()` method is used. The class with the highest probability is taken as the predicted class.

The example in the code predicts which class a specific test image (selected at index `789`) belongs to and prints both the **predicted class** and **prediction probability**.

---

## How to Run the Code

1. **Clone the repository** or copy the code into your environment.
2. **Install dependencies** using pip (details below).
3. Run the script to automatically:
   - Load the dataset.
   - Build and train the CNN model.
   - Evaluate the model on the test dataset.
   - Generate plots for training/validation loss and accuracy over epochs.
   - Predict a sample image category.

---

## Dependencies

To run this script, you need the following libraries installed:

- **Python** (3.6+)
- **TensorFlow** (2.0+)
- **NumPy**
- **matplotlib**

You can install the required libraries using pip:

```bash
pip install tensorflow numpy matplotlib
```

---
