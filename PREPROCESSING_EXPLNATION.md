

# PlantVillage Dataset Preprocessing

## Introduction

This repository contains code for preprocessing the PlantVillage dataset. The PlantVillage dataset is a large dataset of labeled images consisting of healthy and diseased plant leaves. Preprocessing is a crucial step in preparing the dataset for training machine learning models, particularly for tasks like image classification. This preprocessing script performs various tasks such as image segmentation, morphological processing, data augmentation, and creating data generators for training, validation, and testing.

## Code Explanation

### Cloning the Repository

The first step is to clone the PlantVillage dataset repository from GitHub. This ensures that we have access to the dataset for preprocessing.

```python
!git clone https://github.com/spMohanty/PlantVillage-Dataset
%cd PlantVillage-Dataset/raw/color
```

### Installing Necessary Libraries

Ensure that all required libraries are installed by running the following command:

```python
!pip install tensorflow numpy pandas opencv-python scikit-learn
```

### Importing Libraries

Import the necessary Python libraries for image processing, data manipulation, and machine learning.

```python
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
```

### Preprocessing Functions

Define functions for image segmentation and morphological operations:

```python
def apply_threshold(img):
    # Apply binary thresholding to the image
    ...
    return thresh

def apply_morphology(img):
    # Apply morphological closing operation to the image
    ...
    return morph

def preprocess_image(img, img_size=(128, 128)):
    # Resize the image and apply preprocessing operations
    ...
    return morph_img
```

### Data Generator with Preprocessing

Create a generator function to load images in batches with preprocessing and augmentation:

```python
def data_generator(data_dir, categories, batch_size=32, img_size=(128, 128), augment=True):
    # Initialize the ImageDataGenerator for data augmentation
    ...
    while True:
        data = []
        labels = []
        for category in categories:
            path = os.path.join(data_dir, category)
            class_num = categories.index(category)
            for img in os.listdir(path):
                try:
                    # Load and preprocess the image
                    ...
                    # Append the processed image and label to the data and labels lists
                    ...
                    if len(data) >= batch_size:
                        # Convert data and labels to numpy arrays and yield batches of data
                        ...
                except Exception as e:
                    print(f"Error loading image {img}: {e}")

        if data:
            # Convert data and labels to numpy arrays and yield batches of data
            ...
```

### Initializing Variables and Creating Generators

Initialize variables and create data generators for training, validation, and testing:

```python
data_dir = '/content/PlantVillage-Dataset/raw/color'
categories = os.listdir(data_dir)

img_size = (128, 128)
batch_size = 32

# Create train, validation, and test generators
train_generator = data_generator(data_dir, categories, batch_size=batch_size, img_size=img_size, augment=True)
val_generator = data_generator(data_dir, categories, batch_size=batch_size, img_size=img_size, augment=False)
test_generator = data_generator(data_dir, categories, batch_size=batch_size, img_size=img_size, augment=False)
```

## Conclusion

This README provides an overview of the code for preprocessing the PlantVillage dataset. By following the steps outlined above, you can preprocess the dataset and prepare it for training machine learning models for tasks like plant disease classification.
