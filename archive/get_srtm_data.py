import geopandas as gpd 
import numpy as np 
import pyproj
import pandas as pd 
from shapely import geometry
import elevation
import os
import subprocess
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, Resampling
import rasterio
import netCDF4
from osgeo import gdal
import pickle
import matplotlib.pyplot as plt


# # lat_lon_time = pd.read_csv('lat_lon_time.csv')
# lat_len = 412
# lon_len = 424
# fn = 'pr_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CLMcom-CCLM4-8-17_v1_day_19500101-19501231.nc'
# nc = netCDF4.Dataset(fn, mode='r')
# lat = nc.variables['lat'][:]
# print(lat.shape)
# lon = nc.variables['lon'][:]
# lat_flat = np.reshape(lat, (-1,))
# lon_flat = np.reshape(lon, (-1,))
# # print(lat.shape)
# # print(412*424)
# # geometries = [geometry.Point(lo, la) for lo, la in zip(lon_flat, lat_flat)]
# # multipoints = geometry.MultiPoint(geometries)
# # bounds = multipoints.envelope
# # gpd.GeoSeries(bounds).to_file('./eur_bounds.gpkg', 'GPKG')
# # bounds = gpd.read_file('./data/processed/area_of_study_bounds.gpkg').bounds
# # west, south, east, north = bounds = bounds.loc[0]
# # west, south, east, north = bounds  = west - .05, south - .05, east + .05, north + .05
# # dem_path = 'gt30/dem.tif'
# # dem_raster = rasterio.open('.' + dem_path)
# # src_crs = dem_raster.crs
# # src_shape = src_height, src_width = dem_raster.shape
# # src_transform = from_bounds(west, south, east, north, src_width, src_height)
# # source = dem_raster.read(1)

# driver = gdal.GetDriverByName('GTiff')
# filename = os.getcwd() + '/gt30/dem.tif' #path to raster
# dataset = gdal.Open(filename)
# band = dataset.GetRasterBand(1)

# cols = dataset.RasterXSize
# rows = dataset.RasterYSize

# transform = dataset.GetGeoTransform()

# xOrigin = transform[0]
# yOrigin = transform[3]
# pixelWidth = transform[1]
# pixelHeight = -transform[5]

# data = band.ReadAsArray(0, 0, cols, rows)

# points_list = [(lon_flat[i], lat_flat[i]) for i in range(lat_flat.shape[0])] #list of X,Y coordinates
# dem_output = np.zeros_like(lat_flat)

# for i, point in enumerate(points_list):
#     col = int((point[0] - xOrigin) / pixelWidth)
#     row = int((yOrigin - point[1] ) / pixelHeight)
#     dem_output[i] = data[row][col]
#     if (i + 1) % 500 == 0:
#         print('Done with %d elements.' % i)

# dem_output = dem_output.reshape(lat.shape)
# with open('dem.pkl', 'wb') as f:
#     pickle.dump(dem_output, f)

with open('dem.pkl', 'rb') as f:
    dem = pickle.load(f)
# # dem = np.flipud(dem)
# # with open('dem.pkl', 'wb') as f:
# #     pickle.dump(dem, f)
plt.imshow(dem)
plt.show()

# lat = lat_lon_time['lat'][:lat_len].values
# lon = lat_lon_time['lon'][:lon_len].values
# coord_list = []
# for la in lat:
#   for lo in lon:
#       coord_list.append((lo, la))
# geometries = [geometry.Point(*coord) for coord in coord_list]
# multipoints = geometry.MultiPoint(geometries)
# bounds = multipoints.envelope
# gpd.GeoSeries(bounds).to_file('./eur_bounds.gpkg', 'GPKG')
# bounds = gpd.read_file('./eur_bounds.gpkg').bounds
# west, south, east, north = bounds.loc[0]
# print(np.amin(lon))
# print(np.amin(lat))
# print(np.amax(lon))
# print(np.amax(lat))
# west, south, east, north = bounds  = west - .05, south - .05, east + .05, north + .05
# # print(west)
# # print(south)
# # print(east)
# # print(north)
# num_tiles = 20.0
# delta_ns = (north - south) / num_tiles
# delta_ew = (east - west) / num_tiles
# curr_west = west
# curr_south = south
# for i in range(int(num_tiles)):
#   curr_north = curr_south + delta_ns
#   curr_west = west
#   for j in range(int(num_tiles)):
#       curr_east = curr_west + delta_ew
#       dem_path = '/srtm/eur_srtm_' + str(i) + '_' + str(j) + '.tif'
#       output = os.getcwd() + dem_path
#       subprocess.call(['eio', 'clip', '-o', output, '--bounds', str(curr_west), str(curr_south), str(curr_east), str(curr_north)])
#       # elevation.clip(bounds=(curr_west, curr_south, curr_east, curr_north), output=output, product='SRTM3')
#       # eio clip -o Rome-30m-DEM.tif --bounds 12.35 41.8 12.65 42
#       # curr_bounds = curr_west, curr_south, curr_east, curr_north
#       curr_west = curr_east
#   curr_south = curr_north