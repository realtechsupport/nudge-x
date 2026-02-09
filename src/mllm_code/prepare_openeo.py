# prepare_openeo.py
# Sentinel-2 image quality check
# -*- coding: utf-8 -*-
# Nov 2024
# https://pro.arcgis.com/en/pro-app/3.3/help/analysis/raster-functions/band-arithmetic-function.htm
# May 2025 updated
# -------------------------------------------------------------------------------------------------

import math, csv

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
        
    return (job)
    
# -------------------------------------------------------------------------------------------------
def get_sites(file_path, limit):
    j=0
    bboxes = []
    cities = []
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) # Skip the header row
            for row in csv_reader:
                city = row[0]
                latitude = float(row[2])
                longitude = float(row[3])
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