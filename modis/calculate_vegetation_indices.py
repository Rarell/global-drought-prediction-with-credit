import numpy as np
from glob import glob
from netCDF4 import Dataset
from datetime import datetime
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def calculate_ndvi(red, nir):
    '''
    Calculate the Normalized Difference Vegetation Index using EM refractance from the red and NIR spectrum
    '''

    # Calculate the NDVI
    ndvi = (nir - red)/(nir + red)
    return ndvi

def calculate_evi(blue, red, nir):
    '''
    Calculate the Enhanced Vegetation Index using the EM refractance from the blue, red, and NIR spectrum
    '''

    # Calculate the EVI
    G = 2.5 # Gain
    L = 1 # Background adjustment 
    C1 = 6; C2 = 7.5 # Aerosol resistence terms
    evi = G * (nir - red)/(nir + (C1*red) - (C2*blue) + L)
    return evi

def calculate_lai_and_fpar(red, nir, surface_type):

    # Table for backup algorithm

    return


def test_map(data, lat, lon, date, data_name):
    '''
    Create a test map to check vegetation index calculations and ensure they were done correctly.
    '''

    # Turn bad values into NaNs
    data[data == -9999] = np.nan

    # Lonitude and latitude tick information
    lat_int = 15
    lon_int = 30
    
    lat_label = np.arange(-90, 90, lat_int)
    lon_label = np.arange(-180, 180, lon_int)


    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()

    # Colorbar information
    #### NOTE: Might need to adjust cmin, cmax, and cint for other variables
    cmin = -1; cmax = 1; cint = 0.02
    
    clevs = np.arange(cmin, cmax+cint, cint)
    nlevs = len(clevs) - 1
    cmap  = plt.get_cmap(name = 'BrBG', lut = nlevs)
    
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()

    # Figure
    fig = plt.figure(figsize = [12, 16])
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

    ax.set_title('%s for %s'%(data_name, date.strftime('%Y-%m-%d')), fontsize = 16)    

    # Add coastlines
    ax.coastlines()

    # Set tick information
    ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
    ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
    ax.set_xticklabels(lon_label, fontsize = 14)
    ax.set_yticklabels(lat_label, fontsize = 14)

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    # Plot the data
    data = data.astype(np.float32)
    cs = ax.contourf(lon, lat, data[:,:], levels = clevs, cmap = cmap, 
                     transform = data_proj, extend = 'both', zorder = 1)

    # Add a colorbar
    cbax = fig.add_axes([0.125, 0.30, 0.80, 0.02])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

    # Set an extent for the map
    ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], 
                    crs = fig_proj)
    
    # Save the map
    plt.savefig('%s_%s_test_map.png'%(data_name, date.strftime('%Y-%m-%d')))

    plt.show(block = False)
    
    # Close the figure
    plt.close('all')

def load_nc(filename, path = './'):
    '''
    Load an netCDF file with multiple variables located within it

    The data is stored in a dictionary with the same keys as the netCDF file
    '''

    data = {}

    # Load the data for each variable within it
    with Dataset(path + filename, 'r') as nc:
        for key in nc.variables.keys():
            data[key] = nc.variables[key][:]

    return data

def write_nc(filename, data, lat, lon, var_name, date, path = './'):
    '''
    Write a netCDF file for geospatial data
    '''

    # Make a description for MODIS data
    description = 'Global MODIS %s for an %d-day period starting at %s, and averaged down to 0.05 degree x 0.05 degree resolution.'%(var_name.upper(), 8, date[0].strftime('%b %d, %Y'))

    I, J = data.shape

    T = len(date)

    # Write the file
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = description

        # Create the spatial and temporal dimensions
        nc.createDimension('lat', size = I)
        nc.createDimension('lon', size = J)
        nc.createDimension('time', size = T)

        # Create the lat and lon variables       
        nc.createVariable('lat', lat.dtype, ('lat', 'lon'))
        nc.createVariable('lon', lon.dtype, ('lat', 'lon'))
        
        nc.variables['lat'][:,:] = lat[:,:]
        nc.variables['lon'][:,:] = lon[:,:]

        # Create the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(date)):
            nc.variables['date'][n] = str(date[n])

        nc.createVariable(var_name, data.dtype, ('lat', 'lon'))
        nc.variables[str(var_name)][:,:] = data[:,:]


if __name__ == '__main__':
    # Create a parser to parse path and year (allows for multiple scripts to be run for multiple years at a time)
    description = 'Calculate NDVI and EVI from MODIS reflectance data for a given year'
    parser = ArgumentParser(description = description)
    parser.add_argument('--year', type = int, default = 0, help = 'The year to calculate the EVI and NDVI for (0 = 2000, 1 = 2001, ...)')
    parser.add_argument('--path', type = str, default = './', help = 'Directory path from the current directory to the MODIS reflectance data')

    args = parser.parse_args()

    year = args.year + 2000
    path = args.path

    # Collect all nc files in a year
    filenames = glob('%s/%04d/modis.reflectance.8-day.%04d.*.nc'%(path, year, year), recursive = True)
    filenames = np.sort(filenames)

    # Load land classification for LAI and FPAR

    # Loop through each file
    for file in filenames:
        print(file)
        # Load file and necessary bands
        data = load_nc(file, path)
        data['date'] = datetime.fromisoformat(data['date'][0])
        red = data['sur_refl_b01']; nir = data['sur_refl_b02']; blue = data['sur_relf_b03']

        # Calculate each index
        ndvi = calculate_ndvi(red, nir)
        evi = calculate_evi(blue, red, nir)

        # Remove bad data points
        ndvi[(ndvi < -1) | (ndvi > 1)] = -9999
        evi[(evi < -1) | (evi > 1)] = -9999

        # Save each index
        ndvi_filename = 'modis.ndvi.8-day.%04d.%02d.%02d.nc'%(data['date'].year, data['date'].month, data['date'].day)
        evi_filename = 'modis.evi.8-day.%04d.%02d.%02d.nc'%(data['date'].year, data['date'].month, data['date'].day)

        write_nc(ndvi_filename, ndvi, data['lat'], data['lon'], 'ndvi', [data['date']])
        write_nc(evi_filename, evi, data['lat'], data['lon'], 'evi', [data['date']])

        # Make plots to show and visualize data (also checks calculations)
        # test_map(ndvi, data['lat'], data['lon'], data['date'], 'ndvi')
        # test_map(evi, data['lat'], data['lon'], data['date'], 'evi')
    