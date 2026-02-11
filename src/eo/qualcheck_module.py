# qualcheck_module.py
# Sentinel-2 image quality check
# -*- coding: utf-8 -*-
# Nov 2024

# Required packages
# !pip install gdal
# !pip install geojson
# !pip install scikit-image --upgrade

#### Imports
import os, sys, time, random, itertools, glob, shutil, json
import numpy
from osgeo import gdal
import cv2
from skimage import io, filters, measure
from scipy.stats import entropy
from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from skimage.exposure import histogram
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageChops

import matplotlib.pyplot as plt
import base64
import io as python_io


def normalize_geotiff_RGB(image_path):
    # Open the dataset
    dataset = gdal.Open(image_path)

    if dataset is None:
        print(f"Unable to open {image_path}")
        return None

    # Get dataset information
    bands = dataset.RasterCount
    print(f"Number of bands: {bands}")

    # Check if the file has at least 3 bands (for RGB)
    if bands < 3:
        print("This file doesn't have enough bands for true color display.")
        return None

    # Read the red, green, blue bands
    red = dataset.GetRasterBand(1).ReadAsArray().astype(numpy.float32)
    green = dataset.GetRasterBand(2).ReadAsArray().astype(numpy.float32)
    blue = dataset.GetRasterBand(3).ReadAsArray().astype(numpy.float32)

    # Stack only RGB for visualization
    rgb = numpy.dstack((red, green, blue))
    rgb = rgb.astype(numpy.float32)

    # Normalize RGB for visualization
    rgb_normalized = (rgb - numpy.min(rgb)) / (numpy.max(rgb) - numpy.min(rgb))

    # Return RGB image for visualization
    return (rgb_normalized)


def evaluate_image_quality(image_path):
    # Read the image
    image = io.imread(image_path)

    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        # Convert the image to 8-bit unsigned integer format
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Calculate contrast
    contrast = numpy.std(gray_image)

    # Calculate entropy (mapped to 0-8 range)
    entropy = shannon_entropy(gray_image)

    # Calculate sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = numpy.var(laplacian)

    # Calculate noise using the mean of local standard deviations
    local_std = ndimage.generic_filter(gray_image, numpy.std, size=5)
    noise = numpy.mean(local_std)

    # Normalize scores to a 0-100 scale
    max_contrast = 255  # Maximum possible contrast for 8-bit images
    max_entropy = 8  # Maximum possible entropy for 8-bit images
    max_sharpness = 1000  # This is an arbitrary value, adjust based on your images
    max_noise = 50  # This is an arbitrary value, adjust based on your images

    contrast_score = (contrast / max_contrast) * 100
    entropy_score = (entropy / max_entropy) * 100
    sharpness_score = min((sharpness / max_sharpness) * 100, 100)
    noise_score = max(100 - (noise / max_noise) * 100, 0)

    # Calculate overall quality score (simple average)
    overall_score = (contrast_score + entropy_score + sharpness_score + noise_score) / 4

    return {
        'contrast': contrast_score,
        'entropy': entropy_score,
        'sharpness': sharpness_score,
        'noise': noise_score,
        'overall': overall_score
    }

def generate_thumbnail_RGB(image_path, size=(100, 100)):
    # Use GDAL to open the image
    dataset = gdal.Open(image_path)

    if dataset is None:
        print(f"Unable to open {image_path}")
        return None

    # Check if the file has at least 3 bands (for RGB)
    if dataset.RasterCount < 3:
        print(f"{image_path} does not have enough bands for RGB display.")
        return None

    # Read the RGB bands
    red = dataset.GetRasterBand(1).ReadAsArray().astype(numpy.float32)
    green = dataset.GetRasterBand(2).ReadAsArray().astype(numpy.float32)
    blue = dataset.GetRasterBand(3).ReadAsArray().astype(numpy.float32)

    # Stack RGB bands for display and normalization
    rgb = numpy.dstack((red, green, blue))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255  # Normalize to 0-255 range
    rgb = rgb.astype(numpy.uint8)

    # Convert the RGB array to a PIL image and resize to thumbnail
    img = Image.fromarray(rgb, mode="RGB")
    img.thumbnail(size)
    # Save the thumbnail as a base64 encoded string
    buffer = python_io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f'<img src="data:image/png;base64,{encoded}" width="{size[0]}" height="{size[1]}"/>'



def calculate_metrics(grayscale_image):
    contrast = numpy.std(grayscale_image)
    entropy = shannon_entropy(grayscale_image)
    laplacian = cv2.Laplacian(grayscale_image, cv2.CV_64F)
    sharpness = numpy.var(laplacian)
    local_std = ndimage.generic_filter(grayscale_image, numpy.std, size=5)
    noise = numpy.mean(local_std)
    return (contrast, entropy, sharpness, noise)


def evaluate_image_quality_IR_RGB(image_path):
    dataset = gdal.Open(image_path)
    if dataset is None or dataset.RasterCount < 4:
        print(f"{image_path} does not have enough bands for RGB and IR display.")
        return None

    # Read RGB and IR bands
    red = dataset.GetRasterBand(1).ReadAsArray().astype(numpy.float32)
    green = dataset.GetRasterBand(2).ReadAsArray().astype(numpy.float32)
    blue = dataset.GetRasterBand(3).ReadAsArray().astype(numpy.float32)
    ir = dataset.GetRasterBand(4).ReadAsArray().astype(numpy.float32)

    # Convert RGB and IR bands to grayscale for quality metrics
    rgb_grayscale = (0.2989 * red + 0.5870 * green + 0.1140 * blue).astype(numpy.uint8)
    ir_grayscale = ir.astype(numpy.uint8)

    # Calculate metrics for RGB and IR bands
    rgb_contrast, rgb_entropy, rgb_sharpness, rgb_noise = calculate_metrics(rgb_grayscale)
    ir_contrast, ir_entropy, ir_sharpness, ir_noise = calculate_metrics(ir_grayscale)

    # Normalize scores and calculate overall quality score
    max_contrast, max_entropy, max_sharpness, max_noise = 255, 8, 1000, 50
    contrast_score = ((rgb_contrast + ir_contrast) / (2 * max_contrast)) * 100
    entropy_score = ((rgb_entropy + ir_entropy) / (2 * max_entropy)) * 100
    sharpness_score = min(((rgb_sharpness + ir_sharpness) / (2 * max_sharpness)) * 100, 100)
    noise_score = max(100 - ((rgb_noise + ir_noise) / (2 * max_noise)) * 100, 0)

    overall_score = (contrast_score + entropy_score + sharpness_score + noise_score) / 4
    return {
        'contrast': contrast_score,
        'entropy': entropy_score,
        'sharpness': sharpness_score,
        'noise': noise_score,
        'overall': overall_score
    }

def crop_multiband_tiff(image_path):
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise ValueError(f"Unable to open {image_path}. Ensure it is a valid GeoTIFF file.")

    # Read RGB bands (assuming band order: R=1, G=2, B=3)
    red = dataset.GetRasterBand(1).ReadAsArray()
    green = dataset.GetRasterBand(2).ReadAsArray()
    blue = dataset.GetRasterBand(3).ReadAsArray()

    # Normalize bands and stack into an RGB array
    rgb = numpy.dstack((red, green, blue))
    rgb_normalized = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(numpy.uint8)

    # Convert to a PIL Image
    img = Image.fromarray(rgb_normalized)

    # Trim whitespace
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        cropped.save("cropped_multiband_image.tif")
        return cropped
    return (img)
    
def improve_geotiff_RGB(file_path, improve):
    dataset = gdal.Open(file_path)
    
    if dataset is None:
        print(f"Unable to open {file_path}")
        return None
        
    bands = dataset.RasterCount
    print(f"Number of bands: {bands}")

    red = dataset.GetRasterBand(1).ReadAsArray()
    green = dataset.GetRasterBand(2).ReadAsArray()
    blue = dataset.GetRasterBand(3).ReadAsArray()

    rgb = numpy.dstack((red, green, blue))
    rgb = rgb.astype(numpy.float32)
    rgb_normalized = (rgb - numpy.min(rgb)) / (numpy.max(rgb) - numpy.min(rgb))
    
    result = rgb_normalized

    if(improve == True):
        print("Applying image enhancements..")
        # pick better parameters... 1.5
        rgb_brighter = numpy.clip(rgb_normalized * 1.3 + 0.2, 0, 1)

        # pick a better kernel...
        kernel = numpy.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        rgb_sharpend = cv2.filter2D(rgb_brighter, -1, kernel)
        result = rgb_sharpend
    
    return(result)


