# utilities module
# Dec 2024, RTS
# Updates May 2025

import math
import time
import numpy
import rasterio
from osgeo import gdal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

#------------------------------------------------------------------------------------------
def get_coordinates(latitude,longitude,surface_area):
    R = 6371.0 # Calculate the side length of the square in km
    side_length = math.sqrt(surface_area)
    # Convert side length to degrees (approximation)
    delta_lat = (side_length / R) * (180 / math.pi)
    delta_lon = (side_length / R) * (180 / math.pi) / math.cos(math.radians(latitude))
    top_left = [longitude - delta_lon / 2, latitude + delta_lat / 2]
    bottom_right = [longitude + delta_lon / 2, latitude - delta_lat / 2]
    return (top_left+ bottom_right)
    
#------------------------------------------------------------------------------------------   
def download_relevant_files(assets, filtered_images, selected_coords, folder):
  res = {}
  for i in assets:
    if i.name in filtered_images:
      lat, longt = selected_coords
      lat, longt = abs(int(lat)),abs(int(longt))
      timestamp = int(time.time())
      base_name = i.name.split('.')[0]
      i.name = f"{base_name}_{lat}_{longt}_{timestamp}.tif"
      res[i.name] = i
      i.download(target=folder)
  return (res)

#------------------------------------------------------------------------------------------ 
# visualize a RGB from IR-RGB Sentinel-2 geotif
def process_geotiff(file, bands, datapath ):
    # Open the dataset
    dataset = gdal.Open(datapath + file)

    if dataset is None:
        print(f"Unable to open {file}")
        return None

    # Get dataset information
    num_bands = dataset.RasterCount
    print(f"Number of bands: {num_bands}")

    # Check if the file has at least 3 bands (for RGB)
    if num_bands < 3:
        print("This file doesn't have enough bands for true color display.")
        return None

    # Read the red, green, and blue bands
    # Sentinel-2 bands: 4 (Red), 3 (Green), 2 (Blue)
    arr = []
    for i in bands:
      if i not in range(1,num_bands+1):
        print(f"Invalid band number: {i}")
        return None
      band = dataset.GetRasterBand(i).ReadAsArray().astype(numpy.float32)
      arr.append(band)

    # Stack bands directly
    rgb = numpy.dstack(arr)
    # Normalize the data
    rgb_normalized = (rgb - numpy.min(rgb)) / (numpy.max(rgb) - numpy.min(rgb))

    return(rgb_normalized)
#------------------------------------------------------------------------------------------

def adjust_brightness_contrast_npy(dataset, bands, brightness, contrast):
    """
    Adjust brightness and contrast of the image

    Parameters:
    image: 3-band normalized numpy array generated from .tif (process_tif)
    brightness: Brightness factor (>1 increases brightness, <1 decreases brightness)
    contrast: Contrast factor (>1 increases contrast, <1 decreases contrast)
    """
    # Adjust brightness
    adjusted = dataset * brightness

    # Adjust contrast
    adjusted = (adjusted - 0.5) * contrast + 0.5

    # Clip values to 0-1 range
    adjusted = numpy.clip(adjusted, 0, 1)
    return (adjusted)
#------------------------------------------------------------------------------------------
# added May 2025
# unsharp masking for sharpening
def sharpen(image, sigma=1, strength=1.5):
    blurred = gaussian_filter(image, sigma=sigma)
    return numpy.clip(image + strength * (image - blurred), 0, 1)

# contrast
def contrast_stretch(band, lower_percent=2, upper_percent=98):
    p_low, p_high = numpy.percentile(band, (lower_percent, upper_percent))
    band_clipped = numpy.clip(band, p_low, p_high)
    stretched = (band_clipped - p_low) / (p_high - p_low)
    return (stretched)

# gamma correction
def apply_gamma(image, gamma=1.2):
    return numpy.power(image, gamma)

# brightness boost
def boost_brightness(image, factor=1.2):
    return numpy.clip(image * factor, 0, 1)

# adjust saturation
def adjust_saturation(rgb, factor=0.5):
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] *= factor  # Reduce saturation <1.0
    return (hsv_to_rgb(hsv))
#------------------------------------------------------------------------------------------

def adjust_coloration(file_path, job_title, stretch_contrast, brightness_factor, saturation_factor):
    with rasterio.open(file_path) as src:
        if("rgb" in job_title):
            r = src.read(1).astype(numpy.float32)
            g = src.read(2).astype(numpy.float32)
            b = src.read(3).astype(numpy.float32)

            if(stretch_contrast):
                # Apply contrast stretching
                r = contrast_stretch(r)
                g = contrast_stretch(g)
                b = contrast_stretch(b) 
                
            # Apply gamma
            r = apply_gamma(r)
            g = apply_gamma(g)
            b = apply_gamma(b)
            # Stack into RGB
            rgb = numpy.dstack((r, g, b))
            # Brightness
            rgb = boost_brightness(rgb, brightness_factor)  #1.2
            # Sharpen
            rgb = sharpen(rgb)
            # Saturation
            rgb = adjust_saturation(rgb, saturation_factor)  #0.7
            
        else:
            gray = src.read(1).astype(numpy.float32)
            # Normalize grayscale to [0,1]
            gray_min, gray_max = numpy.min(gray), numpy.max(gray)
            if gray_max > gray_min:
                gray_norm = (gray - gray_min) / (gray_max - gray_min)
            else:
                gray_norm = numpy.zeros_like(gray)

            # contrast stretch ... this may exaggerate results...see Tirana example
            if(stretch_contrast):
                gray_norm = contrast_stretch(gray_norm)

            if("ndvi" in job_title):
                cmap = plt.get_cmap("RdYlGn")   # Red → Yellow → Green
            elif("nbr" in job_title):
                cmap = plt.get_cmap("RdBu")     # Red → Blue
            elif ("ndbi" in job_title):
                built_up = [
                    [1, 1, 1],                  # White
                    [0.7, 0.7, 0.7],
                    [0.6, 0.6, 0.6],
                    [0.5, 0.5, 0.5],
                    [0.4, 0.4, 0.4],
                    [0.6, 0.3, 0.1],            # Brown
                    ]
                cmap = ListedColormap(built_up)
            elif ("fmi" in job_title):
                #cmap = plt.get_cmap("RdGy_r")  # Red Gray    
                cmap = plt.get_cmap("copper")   # copper 

            else:
                cmap = plt.get_cmap("gray")

            # Map grayscale to RGB
            rgb = cmap(gray_norm)[..., :3]  # discard alpha
            
        return(rgb)
#------------------------------------------------------------------------------------------