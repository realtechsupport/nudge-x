# prepare_openeo.py
# Sentinel-2 image quality check
# -*- coding: utf-8 -*-
# Nov 2024
# https://pro.arcgis.com/en/pro-app/3.3/help/analysis/raster-functions/band-arithmetic-function.htm
# May, July, Oct/ Nov 2025 updated
# -------------------------------------------------------------------------------------------------

import os, math, csv
import numpy
import rasterio
from skimage import exposure
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------
def get_coordinates(latitude,longitude,surface_area):
    R = 6371.0 # Calculate the side length of the square in km
    side_length = math.sqrt(surface_area)
    # Convert side length to degrees (approximation)
    delta_lat = (side_length / R) * (180 / math.pi)
    delta_lon = (side_length / R) * (180 / math.pi) / math.cos(math.radians(latitude))
    top_left = [longitude - delta_lon / 2, latitude + delta_lat / 2]
    bottom_right = [longitude + delta_lon / 2, latitude - delta_lat / 2]
    return (top_left+ bottom_right)
    
# ------------------------------------------------------------------------------------------------- 
def create_job_type(connection, bbox, start, end, satellite, band_selection, max_cloud, job_title, aspect):
    cube = connection.load_collection(
    satellite,
    spatial_extent = {"west": min(bbox[2],bbox[0]), "south": min(bbox[3],bbox[1]),"east": max(bbox[0],bbox[2]),"north": max(bbox[3],bbox[1])},
    temporal_extent = [start, end],
    bands = band_selection,
    max_cloud_cover = max_cloud,)
    
    if(aspect == "rgb"):
        #Visible image R, G, B
        cube = cube.filter_bands(["B04", "B03", "B02"])
        cube = cube.save_result(format="GTiff")
        job = cube.create_job(title = job_title)
    
    elif(aspect  == "ndvi"):
        #Normalized difference vegetation index
        red = cube.band("B04")
        nir = cube.band("B08")
        ndvi = (nir - red) / (nir + red)
        ndvi = ndvi.save_result(format="GTiff")
        job = ndvi.create_job(title = job_title)
        
    elif(aspect == "nbr"):
        #Normalized Burn Ratio
        nir = cube.band("B08")
        swir = cube.band("B12")
        nbr = (nir - swir) / (nir + swir)
        nbr = nbr.save_result(format="GTiff")
        job = nbr.create_job(title = job_title)
        
    elif(aspect  == "fmi"):
        #Ferrous Mineral Index
        nir = cube.band("B08")
        swir = cube.band("B12")
        fmi = swir / nir
        fmi = fmi.save_result(format="GTiff")
        job = fmi.create_job(title = job_title)
        
    elif(aspect  == "ndbi"):
        #Normalized Built-up Index
        nir = cube.band("B08")
        swir = cube.band("B12")
        ndbi = (swir - nir) / (swir + nir)
        ndbi = ndbi.save_result(format="GTiff")
        job = ndbi.create_job(title = job_title)
        
    elif(aspect  == "ndbi_combo"):
        #Built-up Index for mining - red, nir, swir
        ndbi_combo = cube.filter_bands(["B04", "B08", "B12"])
        ndbi_combo = ndbi_combo.save_result(format="GTiff")
        job = ndbi_combo.create_job(title = job_title)
        
    elif(aspect  == "ndbi_rgb"):
        #Built-up Index for mining - green, blue, red, nir, swir
        ndbi_rgb = cube.filter_bands(["B02", "B03", "B04", "B08", "B12"])
        ndbi_rgb = ndbi_rgb.save_result(format="GTiff")
        job = ndbi_rgb.create_job(title = job_title)
        
    elif(aspect  == "urban_mining"):
        #Built-up Index for mining - max options
        urban_mining = cube.filter_bands(["B02", "B03", "B04", "B05", "B06", "B07","B08", "B8A", "B11", "B12"])
        urban_mining = urban_mining.save_result(format="GTiff")
        job = urban_mining.create_job(title = job_title)
        
    return (job)
    
# -------------------------------------------------------------------------------------------------
def get_sites_org(file_path, category, limit):
    j=0
    bboxes = []
    cities = []
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) # Skip the header row
            for row in csv_reader:
                city = row[0]
                if(category == "cities"):
                    latitude = float(row[2])
                    longitude = float(row[3])
                else:
                    latitude = float(row[3])
                    longitude = float(row[4])
                    
                bbox = get_coordinates(latitude,longitude,surface_area=100)
                #print(city, bbox)
                if(j < limit):
                    #print(city, bbox)
                    bboxes.append(bbox)
                    cities.append(city)
                    j+=1
                else:
                    break
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")

    return(cities, bboxes)
    
# -------------------------------------------------------------------------------------------------   
def get_sites(file_path, category, delimiter, limit):
    j=0
    bboxes = []
    cities = []
    try:
        with open(file_path, 'r') as file:
            if(delimiter == '\t'):
                reader = csv.reader(file, delimiter='\t')
            else:
                reader = csv.reader(file)
                
            next(reader) # Skip the header row
            
            for col in reader:
                city = col[0]
                if(category == "cities"):
                    latitude = float(col[2])
                    longitude = float(col[3])
                else:
                    lat_long = col[5]
                    if(lat_long == ''):
                        latitude = float(col[3])
                        longitude = float(col[4])
                    else:
                        latitude = float(lat_long.split(',')[0])
                        longitude = float(lat_long.split(',')[1])
                        
                bbox = get_coordinates(latitude,longitude,surface_area=100)
                if(j < limit):
                    bboxes.append(bbox)
                    cities.append(city)
                    j+=1
                else:
                    break
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")

    return(cities, bboxes)
# -------------------------------------------------------------------------------------------------
def recompile(metadata):
  recompiled_metadata = {}
  for k in metadata['assets'].keys():
    recompiled_metadata[k] = {i['name']:i['statistics']['valid_percent'] for i in metadata['assets'][k]['raster:bands']}
  return recompiled_metadata

# -------------------------------------------------------------------------------------------------

# Use Bands to filter!! like IR, then RGB, etc
def filter(lists, threshold=96.5):
  res = []
  for k,stats in lists.items():
    if min(stats.values()) > threshold:
      res.append(k)
  return (res)
  
# -------------------------------------------------------------------------------------------------
# Function to extract the RGB bands from a multi-band Sentinel-2 geotif that contains RGB and other bands
def create_rgb_png(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            return
    
        with rasterio.open(input_path) as src:
            # We need to read them in R-G-B order (B04, B03, B02)
            red = src.read(3)
            green = src.read(2)
            blue = src.read(1)
            
            # Get metadata to ensure dimensions match later if needed
            profile = src.profile
    
        # Stack the single bands into a 3D array (Height, Width, 3)
        rgb_stack = numpy.dstack((red, green, blue))
    
        # Raw Sentinel-2 data is usually uint16 (0-65535) or similar. Standard PNGs need uint8 (0-255).
        # Use a 2% and 98% percentile stretch to remove outliers and brighten the image
        p2, p98 = numpy.percentile(rgb_stack, (2, 98))
        
        # Rescale intensity using scikit-image. Matplotlib wants floats 0-1 or ints 0-255
        rgb_rescaled = exposure.rescale_intensity(rgb_stack, in_range=(p2, p98), out_range=(0, 1))
    
        # --- Saving --- Origin='upper' ensures the image isn't flipped vertically
        plt.imsave(output_path, rgb_rescaled, origin='upper')
        return (1)

    except Exception as e:
        return (0)

 # -------------------------------------------------------------------------------------------------  
