# OpenEO Sentinel-2 data collection + operation
#---------------------------------------------------------------------
import os
import openeo
from openeo import *
import json
import numpy
import PIL
import time, random, itertools, shutil

from shapely.geometry import shape
from shapely.ops import unary_union

datapath = "/home/rts/dev/data/" 	# set the data path

#from openeo_helper import initialize, create_job
from openeo_helper import *
from quality_check import recompile, filter
from utilities import *

# normalized burn index
# https://www.mdpi.com/2073-445X/12/2/379
#---------------------------------------------------------------------
# set the variables

# Islamabad
#latitude =  33.73805
#longitude = 73.08449

# Paris
#latitude =  48.86472
#longitude =  2.34901

# London
latitude = 51.5072
longitude = 0.1276

# Los Angeles
#latitude = 34.0549
#longitude = 118.2426

#Lagos
#latitude = 6.5244
#longitude = 3.3792

#Bagdad
#latitude = 33.3152
#longitude = 44.3661

#Tijuana
#latitude = 32.5332
#longitude = 117.0193

#Varanasi
#latitude = 25.3176
#longitude = 82.9739

#Xingtai
#latitude = 37.0705
#longitude = 114.5044

#Linfen
#latitude = 36.099041
#longitude = 111.537889

#Fimiston Super Pit
#latitude = -30.779745
#longitude = 121.4944486

#Damascus
latitude = 33.5111
longitude = 36.3103

#------------------------------------------------------------------------
site = str(latitude) + "_" + str(longitude)
savepath = os.path.join(datapath, site) 

# prevent errors on retry with new settings for same location
try:
	os.mkdir(savepath)
except:
	print("Directory already exists.")

satellite = "SENTINEL2_L2A"
max_cloud = 25
start = "2025-01-15"
end = "2025-02-03" 
#band_selection = ["B04", "B03", "B02"] #RGB, TCI
band_selection = ["B12", "B11", "B08", "B04", "B03", "B02"] 


#sformat = "PNG" 
sformat = "GTiff"

if(sformat == "PNG"):
	stype = "png"
else:
	stype = "tif"

job_title = "idealcityfolly"

#----------------------------------------------------------------------------------------
bbox = get_coordinates(latitude,longitude,surface_area=100)
print("input bounding box: ", bbox)
#----------------------------------------------------------------------------------------   

# authenticate - auth file has your secrets
print("authenticating...")
f = open('auth.txt', 'r')
data = f.read()
jdata = json.loads(data)
f.close()

copernicus_url = 'openeo.dataspace.copernicus.eu'
connection = openeo.connect(url=copernicus_url)

connection.authenticate_oidc_client_credentials(
    client_id = jdata['client_id'],
    client_secret = jdata['client_secret'],
)
#----------------------------------------------------------------------------------------

cube = create_cube_simple(connection,bbox,start,end,satellite, band_selection, max_cloud)

#collection = ["rgb", "ndvi", "ndbi", "fmr", "nbr"]

collection = ["rgb"]

#Sentinel-2a
#https://gisgeography.com/senintel-2-bands-combination/
#B04: red
#B08: near infrared
#B11: swir1 (1610nm)

#NDVI : (nir - red) / (nir + red)
#NDBI : (swir1 - nir) / (swir1 + nir)
#FMR : 	red / swir1
#NBR : 	(nir - swir1) / (nir +  swir1)


for c in collection:
	
	if (c == "ndvi"):
		#NDVI
		job_title = c
		B04 = cube.band("B04")
		B08 = cube.band("B08")
		cube_ndvi = (B08 - B04) / (B08 + B04)
		cube = cube_ndvi.save_result(format = sformat)
	elif (c == "nbdi"):
		#NDBI
		job_title = c
		B11 = cube.band("B11")
		B08 = cube.band("B08")
		cube_ndbi = (B11 - B08) / (B11 + B08)
		cube = cube_ndbi.save_result(format = sformat)
	elif (c == "fmr"):
		# FMR - Ferrous Minerals
		job_title = c
		B04 = cube.band("B04")
		B11 = cube.band("B11")
		cube_fmr = B04 / B11
		cube = cube_fmr.save_result(format = sformat)
	elif (c == "nbr"):
		# NBR - Normalized Burn Index
		job_title = c
		B08 = cube.band("B08")
		B11 = cube.band("B11")
		cube_nbr = (B08 - B11) / (B08 + B11)	
		cube = cube_nbr.save_result(format = sformat)
	elif (c == "rgb"):
		B02 = cube.band("B02")
		B03 = cube.band("B03")
		B04 = cube.band("B04")
		#cube_rgb = cube.merge_cubes(B04, B03, B02)
		#cube_rgb = cube.band_arithmetic("array_concat([B04, B03, B02])")
		#cube_rgb = openeo.processes.merge_cubes(B04, B03, B02)
		#cube = [B04, B03, B02]
		#cube = cube_rgb.save_result(format = sformat)
		cube_rgb = openeo.processes.array_element([B04, B03, B02]).linear_scale_range(0,3000,0,255)
		cube = cube_rgb.save_result(format = sformat)

	print("Performing: ", job_title)
	job = cube.create_job(title = job_title)
	
	try:
		job.start_and_wait()
	except:
		print("something went wrong")

	print("getting results")
	try:
		results = job.get_results()
		assets = results.get_assets()
		metadata = results.get_metadata()
		compiled = recompile(metadata)
		filtered_images = filter(compiled)
		print(filtered_images)
	
		print("downloading the filtered results")
		download_relevant_files_simple(assets, filtered_images, savepath, stype)

		#fold into if else above...
		if(job_title == "ndvi"):
			identifier = "ndvi_"
			cs = "RdYlGn"
			thresh = 0.30
			
		elif(job_title == "ndbi"):
			identifier = "ndbi_"
			cs = "copper"
			thresh = 0.10
			
		elif(job_title == "fmr"):
			identifier = "fmr_"
			cs = "Reds"		
			thresh = 0.80	
		
		elif(job_title == "nbr"):
			identifier = "nbr_"
			cs = "Reds"		#update...
			thresh = 0.80	#update ...
				
		elif(job_title == "rgb"):
			identifier = "rgb_"
			cs = ''
		else:
			identifier = ''
			cs = ''

		for item in filtered_images:
			if(job_title == "rgb"):
				print("converting to png")
				output_png = identifier + item.split('.tif')[0] + '.png'
				tiff_to_png_rasterio(savepath + '/' +  item, savepath + '/' + output_png)
			else:
				print("converting to png with color scale")
				output_png = identifier + item.split('.tif')[0] + '_chatgpt.png'
				convert_geotiff_to_png_chatgpt(savepath + '/' +  item, savepath + '/' + output_png, color_scale = cs)

				output_png = identifier + item.split('.tif')[0] + '_anthropic.png'
				convert_geotiff_to_png_anthropic(savepath + '/' +  item, savepath + '/' + output_png,  index_type=job_title.capitalize(), threshold = thresh)
			
	except:
		print("Something went wrong...")
		exit()

