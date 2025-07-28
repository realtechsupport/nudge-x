# OpenEO Sentinel-2 data collection + operation
#---------------------------------------------------------------------
import os
import openeo
import json
import numpy
import PIL
import time, random, itertools, shutil

from shapely.geometry import shape
from shapely.ops import unary_union

datapath = "/home/rts/dev/data/" 	# set the data path

from openeo_helper import initialize, create_job
from quality_check import recompile, filter
from utilities import *

#---------------------------------------------------------------------
# set the variables
# London
#latitude = 51.5072
#longitude = 0.1276

# Islamabad
#latitude =  33.73805
#longitude = 73.08449

# Paris
#latitude =  48.86472
#longitude =  2.34901

# London
#latitude = 51.5072
#longitude = 0.1276

# Los Angeles
#latitude = 34.01877
#longitude = 118.33757

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

#-----------------------------------------------------------------------

site = str(latitude) + "_" + str(longitude)
savepath = os.path.join(datapath, site) 

# prevent errors on retry with new settings for same location
try:
	os.mkdir(savepath)
except:
	print("Directory already exists.")

satellite = "SENTINEL2_L2A"
max_cloud = 15
start = "2024-12-30"
end = "2025-02-05" 
band_selection = ["B04", "B03", "B02"] #RGB, TCI
#band_selection = ["B08", "B04", "B03", "B02"] #IR R G B

#sformat = "PNG" 
sformat = "GTiff"

if(sformat == "PNG"):
	stype = "png"
else:
	stype = "tif"

job_title = "idealcityfolly"

#---------------------------------------------------------------------
bbox = get_coordinates(latitude,longitude,surface_area=100)
print("input bounding box: ", bbox)
#---------------------------------------------------------------------    

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
#---------------------------------------------------------------------
print("launching openeo job")
job = create_job(connection, bbox, start, end, satellite, band_selection, max_cloud, job_title, sformat)
try:
    job.start_and_wait()
except:
    print("something went wrong")
    
#---------------------------------------------------------------------
print("getting results")
try:
	results = job.get_results()
	assets = results.get_assets()

	print("performing early quality check")
	metadata = results.get_metadata()
	compiled = recompile(metadata)
	filtered_images = filter(compiled)
	print(filtered_images)

	print("downloading the filtered results")
	selected_coords = (latitude, longitude)
	download_relevant_files(assets, filtered_images, selected_coords, savepath, stype)

except:
	print("No results available with current settings...")
	exit()

#---------------------------------------------------------------------
