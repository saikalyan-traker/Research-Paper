import os
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
from skimage import exposure
import numpy as np
# Define the directory containing your images
image_directory = "/content/drive/MyDrive/Moon_Images"
# Define a list of specific image enhancement algorithms and their titles
(10 algorithms)
enhancement_algorithms = [
("Gaussian Filter", ImageFilter.GaussianBlur(radius=2)),
("Median Filter", ImageFilter.MedianFilter(size=3)),
("Contrast Enhancement", "Contrast"),
("Histogram Equalization", "Histogram Equalization"),
("Unsharp Masking", "Unsharp Masking"),
("Gamma Correction", "Gamma Correction"),
("Adaptive Histogram Equalization", "Adaptive Histogram Equalization"),
("Saturation Adjustment", "Saturation Adjustment"),
("Decorrelation Stretch", "Decorrelation Stretch"),
]
# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_directory) if
f.endswith((".jpg", ".jpeg", ".png"))]
# Define the number of rows and columns for the plot grid
num_rows = 3
num_columns = 5
# Function for applying gamma correction
def apply_gamma_correction(image, gamma):
return image.point(lambda x: x ** gamma)
# Function for applying adaptive histogram equalization
def apply_adaptive_histogram_equalization(image, clip_limit=2.0):
img_array = np.array(image)
img_equalized = exposure.equalize_adapthist(img_array,
clip_limit=clip_limit)
return Image.fromarray((img_equalized * 255).astype(np.uint8))
# Function for applying decorrelation stretch
def apply_decorrelation_stretch(image):
img_array = np.array(image)
img_shape = img_array.shape
# Reshape the image data into a 2D array (one row per pixel)
img_reshaped = img_array.reshape(-1, 3)
# Perform decorrelation stretch (Replace with your specific
implementation)
stretched_image = img_reshaped # Placeholder for the decorrelation
stretch
# Reshape the stretched data back to the original shape
stretched_image = stretched_image.reshape(img_shape)
# Convert the stretched data back to image format
stretched_image = Image.fromarray(stretched_image.astype(np.uint8))
return stretched_image
# Loop through each algorithm
for algorithm_name, algorithm in enhancement_algorithms:
# Create a new figure for the current algorithm
fig = plt.figure(figsize=(15, 10))
for col, image_file in enumerate(image_files):
image_path = os.path.join(image_directory, image_file)
original_image = Image.open(image_path)
# Apply the current algorithm to the original image
if isinstance(algorithm, ImageFilter.Filter):
enhanced_image = original_image.filter(algorithm)
elif algorithm == "Contrast":
enhanced_image =
ImageEnhance.Contrast(original_image).enhance(2.0)
elif algorithm == "Histogram Equalization":
enhanced_image = ImageOps.equalize(original_image)
elif algorithm == "Saturation Adjustment":
enhanced_image =
ImageEnhance.Color(original_image).enhance(0.5)
elif algorithm == "Gamma Correction":
enhanced_image = apply_gamma_correction(original_image,
gamma=1.0)
elif algorithm == "Adaptive Histogram Equalization":
enhanced_image =
apply_adaptive_histogram_equalization(original_image, clip_limit=2.0)
elif algorithm == "Unsharp Masking":
enhanced_image =
original_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
elif algorithm == "Decorrelation Stretch":
enhanced_image = apply_decorrelation_stretch(original_image)
# Plot the enhanced image
ax = fig.add_subplot(num_rows, num_columns, col + 1 + num_columns)
ax.imshow(enhanced_image)
ax.set_title(f"{algorithm_name}\nImage {col + 1}") # Set the title
to include the algorithm name and image number
ax.axis("off")
# Add empty subplots to create gaps
for i in range(2):
ax = fig.add_subplot(num_rows, num_columns, num_rows * num_columns
- i)
ax.axis("off")
plt.tight_layout()
plt.show()# Research-Paper
