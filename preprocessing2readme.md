

---

# Image Preprocessing for PlantVillage Dataset

This README describes the preprocessing steps applied to the PlantVillage dataset for feature extraction and preparation for model training.

## Preprocessing Steps

1. **Histogram Equalization**
2. **K-means Clustering**
3. **Contour Tracing**
4. **Discrete Wavelet Transform (DWT)**
5. **Principal Component Analysis (PCA)**
6. **Gray-Level Co-occurrence Matrix (GLCM)**

### Step 1: Histogram Equalization

Histogram equalization improves the contrast of the images by redistributing the intensity values. This step helps in enhancing the features of the images for better analysis.

- For color images, the histogram equalization is applied to the Y channel of the YUV color space.
- For grayscale images, it is directly applied to the intensity values.

### Step 2: K-means Clustering

K-means clustering is used to segment the images into distinct regions. This step helps in isolating the regions of interest from the background.

- The image pixels are reshaped into a 2D array where each row represents a pixel and columns represent the color channels.
- K-means clustering is performed to segment the image into `k` clusters (default is 3).
- The segmented image is reconstructed using the cluster centers.

### Step 3: Contour Tracing

Contour tracing is used to detect the boundaries of objects in the image. This step helps in identifying the shapes and regions of interest within the image.

- The image is converted to grayscale and blurred using a Gaussian filter.
- Edge detection is performed using the Canny edge detector.
- Contours are found using the `findContours` method.

### Step 4: Discrete Wavelet Transform (DWT)

DWT is applied to extract multi-resolution features from the images. This step decomposes the image into different frequency components.

- The `dwt2` function from PyWavelets is used to perform a 2-level wavelet decomposition.
- The decomposition results in four components: LL (approximation), LH (horizontal detail), HL (vertical detail), and HH (diagonal detail).

### Step 5: Principal Component Analysis (PCA)

PCA is used to reduce the dimensionality of the features while preserving the variance. This step helps in simplifying the dataset for efficient processing.

- The LL component from DWT is reshaped and PCA is applied to extract the principal components.
- The number of principal components is specified by the `n_components` parameter (default is 50).

### Step 6: Gray-Level Co-occurrence Matrix (GLCM)

GLCM is used to extract texture features from the images. This step calculates the spatial relationship between pixel intensities.

- The grayscale image is used to compute the GLCM.
- Texture features such as contrast, dissimilarity, homogeneity, energy, and correlation are extracted from the GLCM.

## Saving the Processed Data

The processed features are saved into a CSV file for use in model training. Additionally, the processed images can be saved into a new directory.

### Example Code for Preprocessing

```python
import cv2
import numpy as np
import pywt
from sklearn.decomposition import PCA
from skimage.feature import greycomatrix, greycoprops
import pandas as pd
import os
from glob import glob

# Define preprocessing functions here...

# Directory paths
dataset_path = 'PlantVillage-Dataset/raw/color'
processed_dir = 'processed_images'
os.makedirs(processed_dir, exist_ok=True)

image_files = glob(os.path.join(dataset_path, '*.jpg'))
preprocessed_data = []

for file in image_files:
    image = cv2.imread(file)
    # Apply preprocessing steps
    image = histogram_equalization(image)
    image = kmeans_clustering(image)
    contours = contour_tracing(image)
    LL, LH, HL, HH = dwt(image)
    image_reshaped = LL.reshape(-1, LL.shape[2])
    pca_features = apply_pca(image_reshaped)
    contrast, dissimilarity, homogeneity, energy, correlation = glcm_features(image)
    
    # Collecting all features
    features = np.hstack((pca_features.flatten(), contrast.flatten(), dissimilarity.flatten(),
                          homogeneity.flatten(), energy.flatten(), correlation.flatten()))
    
    preprocessed_data.append(features)
    
    # Save processed images
    processed_image_path = os.path.join(processed_dir, os.path.basename(file))
    cv2.imwrite(processed_image_path, image)

# Save preprocessed features to CSV
preprocessed_df = pd.DataFrame(preprocessed_data)
preprocessed_df.to_csv('preprocessed_features.csv', index=False)

print("Preprocessed data saved to preprocessed_features.csv")
print("Processed images saved to", processed_dir)
```

## Requirements

- OpenCV
- NumPy
- PyWavelets
- Scikit-learn
- Scikit-image
- Pandas

Install the required packages using pip:

```sh
pip install opencv-python numpy pywavelets scikit-learn scikit-image pandas
```

---

This README provides an overview of the preprocessing steps and the code used to process the PlantVillage dataset. Ensure you have the necessary libraries installed and update the dataset paths as needed.
