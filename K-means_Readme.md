
# K-means for Color Compression in Images

## Project Overview
This project demonstrates the application of the K-means clustering algorithm on a photograph of tulips to compress the colors. By clustering the pixel colors, we can see how different values of k (number of clusters) affect the image. This project provides a hands-on understanding of the K-means algorithm, its underlying theory, and its visual impact on image data.


## Modeling Objective
The goal is to use K-means clustering to reduce the number of colors in a photograph of tulips, exploring how different values of k influence the clustering of pixels and the appearance of the image.

## Dependencies
- Python 3.x
- numpy
- pandas
- matplotlib
- plotly
- scikit-learn

## Data Preparation
The photograph of tulips is read into a numpy array. Each pixel is represented by three values: red (R), green (G), and blue (B), collectively known as RGB values. The image is reshaped so that each row represents a single pixel's RGB values.

### Code to Load and Display Image
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read in the image
img = plt.imread('using_kmeans_for_color_compression_tulips_photo.jpg')

# Display the image and its shape
print(img.shape)  # Output: (320, 240, 3)
plt.imshow(img)
plt.axis('off')
plt.show()
```

## K-means Clustering
### Single Cluster (k=1)
When k=1, the K-means algorithm compresses the image to a single color, which is the average color of all the pixels.

### Code for Single Cluster
```python
# Reshape the image to a 2D array of pixels
img_flat = img.reshape(img.shape[0] * img.shape[1], 3)

# Instantiate and fit the KMeans model
kmeans = KMeans(n_clusters=1, random_state=42).fit(img_flat)

# Replace each pixel with the centroid's RGB value
img_flat1 = img_flat.copy()
img_flat1[:, :] = kmeans.cluster_centers_[0]

# Reshape back to the original image shape
img1 = img_flat1.reshape(img.shape)
plt.imshow(img1)
plt.axis('off')
plt.show()
```

### Multiple Clusters (k=3)
When k=3, the image is compressed into three colors. Each pixel's color is replaced by the RGB value of the nearest centroid.

### Code for Multiple Clusters
```python
# Instantiate and fit the KMeans model
kmeans3 = KMeans(n_clusters=3, random_state=42).fit(img_flat)

# Replace each pixel with the nearest centroid's RGB value
img_flat3 = img_flat.copy()
for i in np.unique(kmeans3.labels_):
    img_flat3[kmeans3.labels_ == i, :] = kmeans3.cluster_centers_[i]

# Reshape back to the original image shape
img3 = img_flat3.reshape(img.shape)
plt.imshow(img3)
plt.axis('off')
plt.show()
```

### Varying Clusters (k=2-10)
The images are clustered using values of k from 2 to 10. The resulting images show how increasing the number of clusters affects the color compression.

### Code for Varying Clusters
```python
# Helper function to plot image grid
def cluster_image_grid(k, ax, img=img):
    img_flat = img.reshape(img.shape[0]*img.shape[1], 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(img_flat)
    new_img = img_flat.copy()
    for i in np.unique(kmeans.labels_):
        new_img[kmeans.labels_ == i, :] = kmeans.cluster_centers_[i]
    new_img = new_img.reshape(img.shape)
    ax.imshow(new_img)
    ax.axis('off')

fig, axs = plt.subplots(3, 3)
fig.set_size_inches(9, 12)
axs = axs.flatten()
k_values = np.arange(2, 11)
for i, k in enumerate(k_values):
    cluster_image_grid(k, axs[i], img=img)
    axs[i].title.set_text('k=' + str(k))
plt.show()
```

## Results
- **k=1**: The image is compressed to a single color, which is the average color of all pixels.
- **k=3**: The image is compressed to three colors, corresponding to the dominant colors in the photograph.
- **k=2-10**: The images show progressively more detail as the number of clusters increases, with diminishing returns after a certain point.

## Conclusion
Using K-means clustering for color compression provides a clear demonstration of how the algorithm works and its effect on real-world data. By compressing the colors in an image, we can see the trade-offs between the number of clusters and the level of detail retained.
