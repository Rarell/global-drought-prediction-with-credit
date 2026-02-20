"""Backup script for plotting.py
Holds some of the older figure and figure formats;
designed to back up to older formats if desired

Since this holds backups/older formats, comments are 
not up to date or follow the same structure as the 
rest of the scripts
"""

import numpy as np
from datetime import datetime, timedelta
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import colorbar as mcolorbar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from path_to_raw_datasets import get_var_shortname

variables = {
    '200 mb u': 'U Wind', '500 mb u': 'U Wind',
    '200 mb v': 'V Wind', '500 mb v': 'V Wind',
    '200 mb z': 'Geopotential', '500 mb z': 'Geopotential',
    '200 mb q_tot': 'Total Specific Humidity', '500 mb q_tot': 'Total Specific Humidity',
    'tair': 'Temperature',
    'sp': 'Pressure',
    'd2m': 'Dewpoint Temperature',
    'tp': 'Precipitation', 'tp_7d': 'Precipitation', 'tp_14d': 'Precipitation', 'tp_30d': 'Precipitation',
    'e': 'Evaporation',
    'pev': 'Potential Evaporation',
    'cvh': 'High Vegetation Coverage',
    'cvl': 'Low Vegetation Coverage',
    'ssr': 'Net Shortwave Radiation',
    'ws': 'Wind Speed',
    'fg10': 'Wind Gusts',
    'swvl1': 'Soil Moisture', 'swvl2': 'Soil Moisture', 'swvl3': 'Soil Moisture', 'swvl4': 'Soil Moisture',
    'fdii1': 'FDII', 'fdii2': 'FDII', 'fdii3': 'FDII', 'fdii4': 'FDII',
    'sesr': 'SESR'
}
full_names = {
    '200 mb u': '200 mb U wind', '500 mb u': '500 mb U wind',
    '200 mb v': '200 mb V wind', '500 mb v': '500 mb V wind',
    '200 mb z': '200 mb geopotential', '500 mb z': '500 mb geopotential',
    '200 mb q_tot': '200 mb total specific humidity', '500 mb q_tot': '500 mb total specific humidity',
    'tair': '2 metre temperature',
    'sp': 'surface pressure',
    'd2m': '2 metre dewpoint temperature',
    'tp': '1 day accumulated precipitation', 
    'tp_7d': '7 day accumulated precipitation', 
    'tp_14d': '14 day accumulated precipitation',
    'tp_30d': '30 day accumulated precipitation',
    'e': '1 day accumulated evaporation',
    'pev': '1 day accumulated potential evaporation',
    'cvh': 'high vegetation coverage',
    'cvl': 'low vegetation coverage',
    'ssr': 'net surface shortwave radiation',
    'ws': '2 metre wind speed',
    'fg10': '10 metre wind gusts',
    'swvl1': '0 - 7 cm soil moisture', 
    'swvl2': '7 - 28 cm soil moisture', 
    'swvl3': '28 - 100 cm soil moisture', 
    'swvl4': '100 - 289 soil moisture',
    'fdii1': '0 - 7 cm FDII', 
    'fdii2': '7 - 28 cm FDII', 
    'fdii3': '28 - 100 cm FDII', 
    'fdii4': '100 - 289 cm FDII',
    'sesr': 'SESR'
}
units = {
    '200 mb u': r'm s$^{-1}$', '500 mb u': r'm s$^{-1}$',
    '200 mb v': r'm s$^{-1}$', '500 mb v': r'm s$^{-1}$',
    '200 mb z': r'm$^2$ s$^{-2}$', '500 mb z': r'm$^2$ s$^{-2}$',
    '200 mb q_tot': r'kg kg$^{-1}$', '500 mb q_tot': r'kg kg$^{-1}$',
    'tair': 'K',
    'sp': 'Pa',
    'd2m': 'K',
    'tp': 'm', 'tp_7d': 'm', 'tp_14d': 'm', 'tp_30d': 'm',
    'e': 'm',
    'pev': 'm',
    'cvh': 'unitless', 'cvl': 'unitless',
    'ssr': r'J m$^{-2}$',
    'ws': r'm s$^{-1}$',
    'fg10': r'm s$^{-1}$',
    'swvl1': r'm$^3$ m$^{-3}$', 'swvl2': r'm$^3$ m$^{-3}$', 'swvl3': r'm$^3$ m$^{-3}$', 'swvl4': r'm$^3$ m$^{-3}$',
    'fdii1': 'unitless', 'fdii2': 'unitless', 'fdii3': 'unitless', 'fdii4': 'unitless',
    'sesr': 'unitless'
}
color_information = { # Most of these need fine tuning
    '200 mb u': {'climits': [-60, 60], 'climits_metric': [3, 20], 'cname': 'PuOr'}, '500 mb u': {'climits': [-40, 40], 'climits_metric': [3, 15], 'cname': 'PuOr'},
    '200 mb v': {'climits': [-35, 35], 'climits_metric': [3, 25], 'cname': 'PuOr'}, '500 mb v': {'climits': [-30, 30], 'climits_metric': [3, 15], 'cname': 'PuOr'},
    '200 mb z': {'climits': [8700, 125000], 'cname': 'RdBu_r'}, '500 mb z': {'climits': [4700, 5800], 'cname': 'RdBu_r'}, # Fine tune
    '200 mb q_tot': {'climits': [0, 1.5e-4], 'climits_metric': [1e-5, 8e-5], 'cname': 'BrBG'}, '500 mb q_tot': {'climits': [0, 0.004], 'climits_metric': [0.0005, 0.0020],'cname': 'BrBG'},
    'tair': {'climits': [210, 330], 'climits_metric': [0.5, 5.5], 'cname': 'RdBu_r'},
    'sp': {'climits': [80000, 105000], 'climits_metric': [1000, 4000], 'cname': 'RdBu_r'},
    'd2m':  {'climits': [200, 320], 'cname': 'BrBG'}, # Fine tune
    'tp': {'climits': [0, 0.05], 'climits_metric': [0.002, 0.025], 'cname': 'Greens'}, 
    'tp_7d': {'climits': [0, 0.10], 'climits_metric': [0.002, 0.025], 'cname': 'Greens'}, # Fine tune
    'tp_14d': {'climits': [0, 0.20], 'climits_metric': [0.002, 0.025], 'cname': 'Greens'}, # Fine tune
    'tp_30d': {'climits': [0, 0.50], 'climits_metric': [0.002, 0.025], 'cname': 'Greens'}, # Fine tune
    'e': {'climits': [-0.005, 0], 'climits_metric': [0.0001, 0.002], 'cname': 'BrBG_r'},
    'pev': {'climits': [-0.009, 0], 'climits_metric': [0.0001, 0.004], 'cname': 'BrBG_r'}, 
    'cvh': {'climits': [0, 1.0], 'cname': 'BrBG'}, # Fine tune
    'cvl': {'climits': [0, 1.0], 'cname': 'BrBG'}, # Fine tune
    'ssr': {'climits': [0, 30000000], 'cname': 'Reds'}, # Fine tune
    'ws':  {'climits': [0, 20], 'cname': 'PuOr'}, # Fine tune
    'fg10': {'climits': [0, 25], 'cname': 'PuOr'}, # Fine tune
    'swvl1': {'climits': [0, 0.8], 'climits_metric': [0.030, 0.095], 'cname': 'BrBG'}, 
    'swvl2': {'climits': [0, 0.8], 'cname': 'BrBG'}, # Fine tune
    'swvl3': {'climits': [0, 0.8], 'cname': 'BrBG'}, # Fine tune
    'swvl4': {'climits': [0, 0.8], 'cname': 'BrBG'}, # Fine tune
    'fdii1': {'climits': [0, 60], 'cname': 'Spectral_r'}, # Fine tune
    'fdii2': {'climits': [0, 60], 'cname': 'Spectral_r'}, # Fine tune
    'fdii3': {'climits': [0, 60], 'cname': 'Spectral_r'}, # Fine tune
    'fdii4': {'climits': [0, 60], 'cname': 'Spectral_r'}, # Fine tune
    'sesr': {'climits': [-3.0, 3.0], 'cname': 'BrBG'} # Fine tune
}
subset_information = {
    'africa': {'map_extent': [-35, 35, 335, 53], 'wspace': 0.18, 'wspace_metric': 0.17}
}


# Function to generate a plot of performance metrics with forecast hour
def plot_metric(metric, y, metric_name, var_name, climatology = None, persistence = None,
                month = 6, year = 2018, path = './Figures/', savename = 'timeseries.png'):
    '''
    Create spegatti plot of metrics with bolded line for average performance

    TODO:
        Make docs for function
        Bugfixes
    '''
    
    if var_name in full_names.keys():
        variable = full_names[var_name]
    else:
        variable = 'Overall'

    I, J = metric.shape

    # Make the average plot
    metric_average = np.nanmean(metric, axis = 0)

    # Make the plot
    fig, ax = plt.subplots(figsize = [12, 8])

    # Set the title
    ax.set_title('%s Forecast Skill vs Forecast Length\nfor Forecasts starting in %d, %d'%(variable, month, year), fontsize = 18)
    
    # Make the plot
    for i in range(I):
        ax.plot(y, metric[i,:], color = 'grey', linewidth = 0.5)

    # Plot Average line
    ax.plot(y, metric_average, color = 'k', linewidth = 2.5, label = 'CrossFormer')

    # Plot climatology and persistence (note these include forecasts at day = 0, so start from 1)
    if climatology is not None:
        ax.plot(y, climatology, color = 'g', linewidth = 2.5, label = 'Climatology')

    if persistence is not None:
        ax.plot(y, persistence, color = 'b', linewidth = 2.5, label = 'Persistence')

    if (climatology is not None) | (persistence is not None):
        ax.legend(fontsize = 18)
        
    # Add labels
    ax.set_ylabel(metric_name.upper(), fontsize = 18)
    ax.set_xlabel('Forecast Day', fontsize = 18)

    # Tick information
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(18)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


# Function to generate maps of true labels and predictions
def make_comparison_map(y, y_pred, climatology, lat, lon, time,
                        var_name = 'tmp', globe = True, forecast_hour = None,
                        path = './Figures/', savename = 'timeseries.png'):
    '''
    Create and save a generic map of y and y_pred

    TODO:
        Add a new column for difference between pred - true
        Update docs
        Bugfixes
    
    Inputs:
    #:param var: 2D array of the variable to be mapped # Add y and y_pred and time and forecast_hour
    :param lat: 2D array of latitudes
    :param lon: 2D array of longitudes
    :param var_name: String. Name of the variable being plotted
    :param globe: Bool. Indicates whether the map will be a global one or not (non-global maps are fixed on the U.S.)
    :param path: String. Output path to where the figure will be saved
    :param savename: String. Filename the figure will be saved as
    '''
    
    # Set colorbar information
    cname = color_information[var_name]['cname']
    cmin = color_information[var_name]['climits'][0]; cmax = color_information[var_name]['climits'][1]; cint = (cmax - cmin)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = cname, lut = nlevs)
    
    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 20
    else:
        lat_int = 30
        lon_int = 60
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.EckertIII()
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    
    title_information = full_names[var_name]
    if forecast_hour is None:
        # labels = ['True %s'%title_information, 'Climatology Prediction for %s'%title_information, '%s Prediction'%title_information]
        labels = ['True %s'%title_information, 'Climatology', '%s'%title_information]
    else:
        # labels = ['True %s'%title_information, 'Climatology Prediction for %s'%title_information, '%d Day %s Prediction'%(forecast_hour/24, title_information)]
        labels = ['True %s'%title_information, 'Climatology', '%d Day %s'%(forecast_hour/24, title_information)]

    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 27], nrows = 3, ncols = 2, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(hspace = -0.75, wspace = 0.4)
    #plt.subplots_adjust(hspace = -0.4)
    for n, data in enumerate([y, climatology, y_pred]):
        # Set the title
        axes[n,0].set_title('%s\nValid: %s'%(labels[n], time.isoformat()), size = 18)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        axes[n,0].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
        
        # Ocean covers and "masks" data outside the U.S.
        axes[n,0].coastlines(edgecolor = 'black', zorder = 3)
        axes[n,0].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        if fig_proj == ccrs.EckertIII():
            gl = axes[n,0].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                   draw_labels = True,
                                   x_inline = False, y_inline = False,
                                   lw = 0.8, linestyle = '--', color = 'grey')
            gl.xlabel_style = {'size': 16}
            gl.ylabel_style = {'size': 16}
            gl.rotate_labels = False
        else:
            axes[n,0].set_xticks(LonLabel, crs = data_proj)
            axes[n,0].set_yticks(LatLabel, crs = data_proj)

            axes[n,0].set_yticklabels(LatLabel, fontsize = 16)
            axes[n,0].set_xticklabels(LonLabel, fontsize = 16)

            axes[n,0].xaxis.set_major_formatter(LonFormatter)
            axes[n,0].yaxis.set_major_formatter(LatFormatter)

        # Plot the data
        cs = axes[n,0].pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                                cmap = cmap, transform = data_proj, zorder = 1)


    # Set the colorbar size and location
    if (var_name == 'ndvi') | (var_name == 'fpar') | (var_name == 'cvh') | (var_name == 'cvl'):
        extend = None
    elif ('tp' in var_name) | ('swvl' in var_name) | ('fdii' in var_name) | ('q_tot' in var_name) | (var_name == 'ws') | (var_name == 'fg10') | (var_name == 'ssr'):
        extend = 'max'
    else:
        extend = 'both'

    if np.invert(globe):
        cbax = fig.add_axes([0.965, 0.30, 0.020, 0.40])
    else:
        cbax = fig.add_axes([0.965, 0.34, 0.020, 0.32])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend  = extend, orientation = 'vertical')
    cbar.ax.set_ylabel('%s (%s)'%(variables[var_name], units[var_name]), fontsize = 18)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(16)
        
        
    # Make the true - pred plot
    cmin_new = anomaly_plotting_information[var_name]['difference_climits'][0]; cmax_new = anomaly_plotting_information[var_name]['difference_climits'][1]
    cint = (cmax_new - cmin_new)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin_new, cmax_new + cint, cint)
    nlevs = len(clevs)
    cmap_new = plt.get_cmap(name = 'coolwarm_r', lut = nlevs)
    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    axes[0,1].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Ocean covers and "masks" data outside the U.S.
    axes[0,1].coastlines(edgecolor = 'black', zorder = 3)
    axes[0,1].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

    # Adjust the ticks
    if fig_proj == ccrs.EckertIII():
        gl = axes[0,1].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                draw_labels = True,
                                x_inline = False, y_inline = False,
                                lw = 0.8, linestyle = '--', color = 'grey')
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        gl.rotate_labels = False
    else:
        axes[0,1].set_xticks(LonLabel, crs = data_proj)
        axes[0,1].set_yticks(LatLabel, crs = data_proj)

        axes[1,1].set_yticklabels(LatLabel, fontsize = 14)
        axes[0,1].set_xticklabels(LonLabel, fontsize = 14)

        axes[0,1].xaxis.set_major_formatter(LonFormatter)
        axes[0,1].yaxis.set_major_formatter(LatFormatter)

    # Plot the data
    cs = axes[0,1].pcolormesh(lon, lat, y - y_pred, vmin = cmin_new, vmax = cmax_new,
                              cmap = cmap_new, transform = data_proj, zorder = 1)


    cbax = fig.add_axes([1.095, 0.34, 0.020, 0.32])#[0.925, 0.235, 0.020, 0.21])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap_new, extend = 'both', orientation = 'vertical')
    cbar.ax.set_ylabel('Difference (%s)'%(units[var_name]), fontsize = 16)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(16)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()

def make_anomaly_map(y, y_pred, lat, lon, time,
                        var_name = 'tmp', globe = True, forecast_hour = None,
                        path = './Figures/', savename = 'timeseries.png'):
    '''
    Create and save a generic map of y and y_pred

    TODO:
        Add a new column for difference between pred - true
        Update docs
        Bugfixes
    
    Inputs:
    #:param var: 2D array of the variable to be mapped # Add y and y_pred and time and forecast_hour
    :param lat: 2D array of latitudes
    :param lon: 2D array of longitudes
    :param var_name: String. Name of the variable being plotted
    :param globe: Bool. Indicates whether the map will be a global one or not (non-global maps are fixed on the U.S.)
    :param path: String. Output path to where the figure will be saved
    :param savename: String. Filename the figure will be saved as
    '''
    
    # Set colorbar information
    cname = color_information[var_name]['cname']
    cmin = anomaly_plotting_information[var_name]['climits'][0]; cmax = anomaly_plotting_information[var_name]['climits'][1]; cint = (cmax - cmin)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = cname, lut = nlevs)
    
    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 20
    else:
        lat_int = 30
        lon_int = 60
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.EckertIII()
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    if np.invert(globe):
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

        CountriesReader = shpreader.Reader(CountriesSHP)

        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
        
    title_information = full_names[var_name]
    if forecast_hour is None:
        labels = ['True %s'%title_information, '%s Prediction'%title_information]
    else:
        labels = ['True %s'%title_information, '%d Day %s Prediction'%(forecast_hour/24, title_information)]

    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 27], nrows = 2, ncols = 2, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(hspace = -0.80, wspace = 0.4)
    # plt.subplots_adjust(hspace = -0.4)
    for n, data in enumerate([y, y_pred]):
        # Set the title
        axes[n,0].set_title('%s\nValid: %s'%(labels[n], time.isoformat()), size = 18)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        axes[n,0].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
        if np.invert(globe):
            # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
            axes[n,0].add_feature(cfeature.STATES)
            axes[n,0].add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
            axes[n,0].add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
        else:
            # Ocean covers and "masks" data outside the U.S.
            axes[n,0].coastlines(edgecolor = 'black', zorder = 3)
            axes[n,0].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        if fig_proj == ccrs.EckertIII():
            gl = axes[n,0].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                   draw_labels = True,
                                   x_inline = False, y_inline = False,
                                   lw = 0.8, linestyle = '--', color = 'grey')
            gl.xlabel_style = {'size': 16}
            gl.ylabel_style = {'size': 16}
            gl.rotate_labels = False
        else:
            axes[n,0].set_xticks(LonLabel, crs = data_proj)
            axes[n,0].set_yticks(LatLabel, crs = data_proj)

            axes[n,0].set_yticklabels(LatLabel, fontsize = 16)
            axes[n,0].set_xticklabels(LonLabel, fontsize = 16)

            axes[n,0].xaxis.set_major_formatter(LonFormatter)
            axes[n,0].yaxis.set_major_formatter(LatFormatter)

        # Plot the data
        cs = axes[n,0].pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                                cmap = cmap, transform = data_proj, zorder = 1)



    # Set the colorbar size and location
    if np.invert(globe):
        cbax = fig.add_axes([0.965, 0.30, 0.020, 0.40])
    else:
        cbax = fig.add_axes([0.965, 0.40, 0.020, 0.20])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = 'both', orientation = 'vertical')
    cbar.ax.set_ylabel('%s Anomaly (%s)'%(variables[var_name], units[var_name]), fontsize = 18)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(16)
        

    # Make the true - pred plot
    cmin_new = anomaly_plotting_information[var_name]['difference_climits'][0]; cmax_new = anomaly_plotting_information[var_name]['difference_climits'][1]
    cint = (cmax_new - cmin_new)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin_new, cmax_new + cint, cint)
    nlevs = len(clevs)
    cmap_new = plt.get_cmap(name = 'coolwarm_r', lut = nlevs)
    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    axes[0,1].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Ocean covers and "masks" data outside the U.S.
    axes[0,1].coastlines(edgecolor = 'black', zorder = 3)
    axes[0,1].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

    # Adjust the ticks
    if fig_proj == ccrs.EckertIII():
        gl = axes[0,1].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                draw_labels = True,
                                x_inline = False, y_inline = False,
                                lw = 0.8, linestyle = '--', color = 'grey')
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        gl.rotate_labels = False
    else:
        axes[0,1].set_xticks(LonLabel, crs = data_proj)
        axes[0,1].set_yticks(LatLabel, crs = data_proj)

        axes[1,1].set_yticklabels(LatLabel, fontsize = 14)
        axes[0,1].set_xticklabels(LonLabel, fontsize = 14)

        axes[0,1].xaxis.set_major_formatter(LonFormatter)
        axes[0,1].yaxis.set_major_formatter(LatFormatter)

    # Plot the data
    cs = axes[0,1].pcolormesh(lon, lat, y - y_pred, vmin = cmin_new, vmax = cmax_new,
                              cmap = cmap_new, transform = data_proj, zorder = 1)


    cbax = fig.add_axes([1.095, 0.40, 0.020, 0.20])#[0.925, 0.235, 0.020, 0.21])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap_new, extend = 'both', orientation = 'vertical')
    cbar.ax.set_ylabel('Difference (%s)'%(units[var_name]), fontsize = 16)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(16)

        
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()

def make_comparison_subset_map(y, y_pred, climatology, lat, lon, time, subset,
                               var_name = 'tmp', forecast_hour = None,
                               path = './Figures/', savename = 'timeseries.png'):
    '''
    Create and save a generic map of y and y_pred for a subsetted region

    TODO:
        Adjust spacing between figures
        Adjust title to include forecast hour/day
        Adjust colorbar
        Update docs
        Bugfixes
    
    Inputs:
    #:param var: 2D array of the variable to be mapped # Add y and y_pred and time and forecast_hour
    :param lat: 2D array of latitudes
    :param lon: 2D array of longitudes
    :param var_name: String. Name of the variable being plotted
    :param globe: Bool. Indicates whether the map will be a global one or not (non-global maps are fixed on the U.S.)
    :param path: String. Output path to where the figure will be saved
    :param savename: String. Filename the figure will be saved as
    '''
    
    # Subset information
    lower_lat = subset_information[subset]['map_extent'][0]; upper_lat = subset_information[subset]['map_extent'][1]
    lower_lon = subset_information[subset]['map_extent'][2]; upper_lon = subset_information[subset]['map_extent'][3]

    # Set colorbar information
    cname = color_information[var_name]['cname']
    cmin = color_information[var_name]['climits'][0]; cmax = color_information[var_name]['climits'][1]; cint = (cmax - cmin)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = cname, lut = nlevs)
    
    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 20
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    
    title_information = full_names[var_name]
    if forecast_hour is None:
        labels = ['True %s'%title_information, 'Climatolgoy prediction', '%s Prediction'%title_information]
    else:
        labels = ['True %s'%title_information, 'Climatology prediction', '%d Day %s Prediction'%(forecast_hour/24, title_information)]

    # Create the figure
    fig, axes = plt.subplots(figsize = [18, 18], nrows = 2, ncols = 3, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(wspace = subset_information[subset]['wspace'], hspace = subset_information[subset]['hspace'])
    for n, data in enumerate([y, climatology, y_pred]):
        # Set the title
        axes[0,n].set_title('%s\nValid for %s'%(labels[n], time.isoformat()), size = 16)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        axes[0,n].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

        # Ocean covers and "masks" data outside the U.S.
        axes[0,n].coastlines(edgecolor = 'black', zorder = 3)
        axes[0,n].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        if fig_proj == ccrs.EckertIII():
            gl = axes[0,n].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                   draw_labels = True,
                                   x_inline = False, y_inline = False,
                                   lw = 0.8, linestyle = '--', color = 'grey')
            gl.xlabel_style = {'size': 14}
            gl.ylabel_style = {'size': 14}
            gl.rotate_labels = False
        else:
            axes[0,n].set_xticks(LonLabel, crs = data_proj)
            axes[0,n].set_yticks(LatLabel, crs = data_proj)

            axes[0,n].set_yticklabels(LatLabel, fontsize = 14)
            axes[0,n].set_xticklabels(LonLabel, fontsize = 14)

            axes[0,n].xaxis.set_major_formatter(LonFormatter)
            axes[0,n].yaxis.set_major_formatter(LatFormatter)

        # Plot the data
        cs = axes[0,n].pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                                cmap = cmap, transform = data_proj, zorder = 1)

        # Set the map extent
        axes[0,n].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])


    # Set the colorbar size and location
    if (var_name == 'ndvi') | (var_name == 'fpar') | (var_name == 'cvh') | (var_name == 'cvl'):
        extend = None
    elif ('tp' in var_name) | ('swvl' in var_name) | ('fdii' in var_name) | ('q_tot' in var_name) | (var_name == 'ws') | (var_name == 'fg10') | (var_name == 'ssr'):
        extend = 'max'
    else:
        extend = 'both'

    # cbax = fig.add_axes([0.925, 0.365, 0.020, 0.26]) # Settings for a single row
    cbax = fig.add_axes(subset_information[subset]['colorbar_coord'][0])#[0.925, 0.535, 0.020, 0.21])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = extend, orientation = 'vertical')
    cbar.ax.set_ylabel('%s (%s)'%(variables[var_name], units[var_name]), fontsize = 16)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(16)
        


    # Make the true - pred plot
    cmin_new = anomaly_plotting_information[var_name]['difference_climits'][0]; cmax_new = anomaly_plotting_information[var_name]['difference_climits'][1]
    cint = (cmax_new - cmin_new)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin_new, cmax_new + cint, cint)
    nlevs = len(clevs)
    cmap_new = plt.get_cmap(name = 'coolwarm_r', lut = nlevs)
    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    axes[1,0].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Ocean covers and "masks" data outside the U.S.
    axes[1,0].coastlines(edgecolor = 'black', zorder = 3)
    axes[1,0].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

    # Adjust the ticks
    if fig_proj == ccrs.EckertIII():
        gl = axes[1,0].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                draw_labels = True,
                                x_inline = False, y_inline = False,
                                lw = 0.8, linestyle = '--', color = 'grey')
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        gl.rotate_labels = False
    else:
        axes[1,0].set_xticks(LonLabel, crs = data_proj)
        axes[1,0].set_yticks(LatLabel, crs = data_proj)

        axes[1,0].set_yticklabels(LatLabel, fontsize = 14)
        axes[1,0].set_xticklabels(LonLabel, fontsize = 14)

        axes[1,0].xaxis.set_major_formatter(LonFormatter)
        axes[1,0].yaxis.set_major_formatter(LatFormatter)

    # Plot the data
    cs = axes[1,0].pcolormesh(lon, lat, y - y_pred, vmin = cmin_new, vmax = cmax_new,
                              cmap = cmap_new, transform = data_proj, zorder = 1)

    # Set the map extent
    axes[1,0].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    cbax = fig.add_axes(subset_information[subset]['colorbar_coord'][1])#[0.925, 0.235, 0.020, 0.21])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap_new, extend = 'both', orientation = 'vertical')
    cbar.ax.set_ylabel('Difference (%s)'%(units[var_name]), fontsize = 16)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(16)
    
        
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()

def make_anomaly_subset_map(y, y_pred, lat, lon, time, subset,
                            var_name = 'tmp', forecast_hour = None,
                            path = './Figures/', savename = 'timeseries.png'):
    '''
    Create and save a generic map of y and y_pred for a subsetted region

    TODO:
        Adjust spacing between figures
        Adjust title to include forecast hour/day
        Adjust colorbar
        Update docs
        Bugfixes
    
    Inputs:
    #:param var: 2D array of the variable to be mapped # Add y and y_pred and time and forecast_hour
    :param lat: 2D array of latitudes
    :param lon: 2D array of longitudes
    :param var_name: String. Name of the variable being plotted
    :param globe: Bool. Indicates whether the map will be a global one or not (non-global maps are fixed on the U.S.)
    :param path: String. Output path to where the figure will be saved
    :param savename: String. Filename the figure will be saved as
    '''

    # Subset information
    lower_lat = subset_information[subset]['map_extent'][0]; upper_lat = subset_information[subset]['map_extent'][1]
    lower_lon = subset_information[subset]['map_extent'][2]; upper_lon = subset_information[subset]['map_extent'][3]

    # Set colorbar information
    cname = 'BrBG' #color_information[var_name]['cname']
    cmin = anomaly_plotting_information[var_name]['climits'][0]; cmax = anomaly_plotting_information[var_name]['climits'][1]
    cint = (cmax - cmin)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = cname, lut = nlevs)
    
    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 20
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    if subset == 'united_states':
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

        CountriesReader = shpreader.Reader(CountriesSHP)

        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
        
    title_information = full_names[var_name]
    if forecast_hour is None:
        labels = ['True %s'%title_information, '%s Prediction'%title_information]
    else:
        labels = ['True %s'%title_information, '%d Day %s Prediction'%(forecast_hour/24, title_information)]

    # Create the figure
    fig, axes = plt.subplots(figsize = [18, 18], nrows = 2, ncols = 2, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(wspace = subset_information[subset]['wspace'], hspace = subset_information[subset]['hspace_anomaly'])
    for n, data in enumerate([y, y_pred]):
        # Set the title
        axes[0,n].set_title('%s\nValid for %s'%(labels[n], time.isoformat()), size = 16)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        axes[0,n].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

        # Ocean covers and "masks" data outside the U.S.
        axes[0,n].coastlines(edgecolor = 'black', zorder = 3)
        axes[0,n].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        if fig_proj == ccrs.EckertIII():
            gl = axes[0,n].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                   draw_labels = True,
                                   x_inline = False, y_inline = False,
                                   lw = 0.8, linestyle = '--', color = 'grey')
            gl.xlabel_style = {'size': 14}
            gl.ylabel_style = {'size': 14}
            gl.rotate_labels = False
        else:
            axes[0,n].set_xticks(LonLabel, crs = data_proj)
            axes[0,n].set_yticks(LatLabel, crs = data_proj)

            axes[0,n].set_yticklabels(LatLabel, fontsize = 14)
            axes[0,n].set_xticklabels(LonLabel, fontsize = 14)

            axes[0,n].xaxis.set_major_formatter(LonFormatter)
            axes[0,n].yaxis.set_major_formatter(LatFormatter)

        # Plot the data
        cs = axes[0,n].pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                                cmap = cmap, transform = data_proj, zorder = 1)

        # Set the map extent
        axes[0,n].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])


    # Set the colorbar size and location
    # cbax = fig.add_axes([0.925, 0.365, 0.020, 0.26]) # Settings for a single row
    cbax = fig.add_axes(subset_information[subset]['colorbar_coord_anomaly'][0])#[0.925, 0.565, 0.020, 0.26])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = 'both', orientation = 'vertical')
    cbar.ax.set_ylabel('%s Anomaly (%s)'%(variables[var_name], units[var_name]), fontsize = 16)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(16)
        
    # Make the true - pred plot
    cmin_new = anomaly_plotting_information[var_name]['difference_climits'][0]; cmax_new = anomaly_plotting_information[var_name]['difference_climits'][1]
    cint = (cmax_new - cmin_new)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin_new, cmax_new + cint, cint)
    nlevs = len(clevs)
    cmap_new = plt.get_cmap(name = 'coolwarm_r', lut = nlevs)
    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    axes[1,0].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Ocean covers and "masks" data outside the U.S.
    axes[1,0].coastlines(edgecolor = 'black', zorder = 3)
    axes[1,0].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

    # Adjust the ticks
    if fig_proj == ccrs.EckertIII():
        gl = axes[1,0].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                draw_labels = True,
                                x_inline = False, y_inline = False,
                                lw = 0.8, linestyle = '--', color = 'grey')
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        gl.rotate_labels = False
    else:
        axes[1,0].set_xticks(LonLabel, crs = data_proj)
        axes[1,0].set_yticks(LatLabel, crs = data_proj)

        axes[1,0].set_yticklabels(LatLabel, fontsize = 14)
        axes[1,0].set_xticklabels(LonLabel, fontsize = 14)

        axes[1,0].xaxis.set_major_formatter(LonFormatter)
        axes[1,0].yaxis.set_major_formatter(LatFormatter)

    # Plot the data
    cs = axes[1,0].pcolormesh(lon, lat, y - y_pred, vmin = cmin_new, vmax = cmax_new,
                              cmap = cmap_new, transform = data_proj, zorder = 1)

    # Set the map extent
    axes[1,0].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    cbax = fig.add_axes(subset_information[subset]['colorbar_coord_anomaly'][1])#[0.925, 0.165, 0.020, 0.26])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap_new, extend = 'both', orientation = 'vertical')
    cbar.ax.set_ylabel('Difference (%s)'%(units[var_name]), fontsize = 16)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(16)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()


def make_error_map(acc, rmse, lat, lon,
                   var_name = 'tmp', globe = True, 
                   path = './Figures/', savename = 'timeseries.png'):
    '''
    Create and save a map of anomaly correlation coefficient (ACC) and root mean square error (RMSE)

    TODO:
        Adjust spacing between figures
        Adjust title to include forecast hour/day
        Adjust colorbar
        Update docs
        Bugfixes
    
    Inputs:
    :param var: 2D array of the variable to be mapped
    :param lat: 2D array of latitudes
    :param lon: 2D array of longitudes
    :param var_name: String. Name of the variable being plotted
    :param globe: Bool. Indicates whether the map will be a global one or not (non-global maps are fixed on the U.S.)
    :param path: String. Output path to where the figure will be saved
    :param savename: String. Filename the figure will be saved as
    '''
    
    # Set colorbar information
    cname = 'Reds'
    cmin_acc = 0.0; cmax_acc = 0.9; cint_acc = (cmax_acc - cmin_acc)/20
    cmin_rmse = color_information[var_name]['climits_metric'][0]; cmax_rmse = color_information[var_name]['climits_metric'][1]; cint_rmse = (cmax_rmse - cmin_rmse)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs_acc = np.arange(cmin_acc, cmax_acc + cint_acc, cint_acc)
    nlevs_acc = len(clevs_acc)
    cmap_acc  = plt.get_cmap(name = cname, lut = nlevs_acc)

    clevs_rmse = np.arange(cmin_rmse, cmax_rmse + cint_rmse, cint_rmse)
    nlevs_rmse = len(clevs_rmse)
    cmap_rmse  = plt.get_cmap(name = cname, lut = nlevs_rmse)
    
    cmins = [cmin_acc, cmin_rmse]; cmaxs = [cmax_acc, cmax_rmse]; cmaps = [cmap_acc, cmap_rmse]

    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 20
    else:
        lat_int = 30
        lon_int = 60
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.EckertIII()
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    if np.invert(globe):
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

        CountriesReader = shpreader.Reader(CountriesSHP)

        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
        
    title_information = full_names[var_name]
    labels = ['ACC for %s predictions'%title_information, 'RMSE for %s predictions'%title_information]
    color_labels = ['ACC', 'RMSE (%s)'%units[var_name]]
    vertical_coords = [0.55, 0.25]

    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 18], nrows = 2, ncols = 1, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(hspace = -0.4)
    for n, data in enumerate([acc, rmse]):
        # Set the title
        axes[n].set_title(labels[n], size = 18)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        axes[n].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
        if np.invert(globe):
            # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
            axes[n].add_feature(cfeature.STATES)
            axes[n].add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
            axes[n].add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
        else:
            # Ocean covers and "masks" data outside the U.S.
            axes[n].coastlines(edgecolor = 'black', zorder = 3)
            axes[n].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        if fig_proj == ccrs.EckertIII():
            gl = axes[n].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                   draw_labels = True,
                                   x_inline = False, y_inline = False,
                                   lw = 0.8, linestyle = '--', color = 'grey')
            gl.xlabel_style = {'size': 16}
            gl.ylabel_style = {'size': 16}
            gl.rotate_labels = False
        else:
            axes[n].set_xticks(LonLabel, crs = data_proj)
            axes[n].set_yticks(LatLabel, crs = data_proj)

            axes[n].set_yticklabels(LatLabel, fontsize = 16)
            axes[n].set_xticklabels(LonLabel, fontsize = 16)

            axes[n].xaxis.set_major_formatter(LonFormatter)
            axes[n].yaxis.set_major_formatter(LatFormatter)

        # Plot the data
        cs = axes[n].pcolormesh(lon, lat, data, vmin = cmins[n], vmax = cmaxs[n],
                                cmap = cmaps[n], transform = data_proj, zorder = 1)

        # Set the map extent to the U.S.
        if np.invert(globe):
            axes[n].set_extent([-130, -65, 23.5, 48.5])
        else:
            axes[n].set_extent([-179, 179, -65, 80])


        # Set the colorbar size and location
        if np.invert(globe):
            cbax = fig.add_axes([0.965, 0.30, 0.020, 0.40])
        else:
            cbax = fig.add_axes([0.965, vertical_coords[n], 0.020, 0.18])
        cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmaps[n], orientation = 'vertical')
        cbar.ax.set_ylabel(color_labels[n], fontsize = 18)

        # Set the colorbar ticks
        for i in cbar.ax.yaxis.get_ticklabels():
            i.set_size(16)
        
        
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_error_subset_map(acc, rmse, lat, lon, subset,
                          var_name = 'tmp', path = './Figures/', savename = 'timeseries.png'):
    '''
    Create and save a Create and save a map of anomaly correlation coefficient (ACC) and root mean square error (RMSE) for a subsetted region

    TODO:
        Adjust spacing between figures
        Adjust title to include forecast hour/day
        Adjust colorbar
        Update docs
        Bugfixes
    
    Inputs:
    :param var: 2D array of the variable to be mapped # Add y and y_pred and time and forecast_hour
    :param lat: 2D array of latitudes
    :param lon: 2D array of longitudes
    :param var_name: String. Name of the variable being plotted
    :param globe: Bool. Indicates whether the map will be a global one or not (non-global maps are fixed on the U.S.)
    :param path: String. Output path to where the figure will be saved
    :param savename: String. Filename the figure will be saved as
    '''
    
    # Subset information
    lower_lat = subset_information[subset]['map_extent'][0]; upper_lat = subset_information[subset]['map_extent'][1]
    lower_lon = subset_information[subset]['map_extent'][2]; upper_lon = subset_information[subset]['map_extent'][3]

    # Set colorbar information
    cname = 'Reds'
    cmin_acc = 0.0; cmax_acc = 0.9; cint_acc = (cmax_acc - cmin_acc)/20
    cmin_rmse = color_information[var_name]['climits_metric'][0]; cmax_rmse = color_information[var_name]['climits_metric'][1]; cint_rmse = (cmax_rmse - cmin_rmse)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs_acc = np.arange(cmin_acc, cmax_acc + cint_acc, cint_acc)
    nlevs_acc = len(clevs_acc)
    cmap_acc  = plt.get_cmap(name = cname, lut = nlevs_acc)

    clevs_rmse = np.arange(cmin_rmse, cmax_rmse + cint_rmse, cint_rmse)
    nlevs_rmse = len(clevs_rmse)
    cmap_rmse  = plt.get_cmap(name = cname, lut = nlevs_rmse)
    
    cmins = [cmin_acc, cmin_rmse]; cmaxs = [cmax_acc, cmax_rmse]; cmaps = [cmap_acc, cmap_rmse]
    
    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 20
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    if subset == 'united_states':
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

        CountriesReader = shpreader.Reader(CountriesSHP)

        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
        
    title_information = full_names[var_name]
    labels = ['ACC for %s predictions'%title_information, 'RMSE for %s predictions'%title_information]
    color_labels = ['ACC', 'RMSE (%s)'%units[var_name]]
    horizontal_coords = [0.132, 0.55]

    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 18], nrows = 1, ncols = 2, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(wspace = subset_information[subset]['wspace_metric'])
    for n, data in enumerate([acc, rmse]):
        # Set the title
        axes[n].set_title(labels[n], size = 16)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        axes[n].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
        if subset == 'united_states':
            # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
            axes[n].add_feature(cfeature.STATES)
            axes[n].add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
            axes[n].add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
        else:
            # Ocean covers and "masks" data outside the U.S.
            axes[n].coastlines(edgecolor = 'black', zorder = 3)
            axes[n].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        if fig_proj == ccrs.EckertIII():
            gl = axes[n].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                   draw_labels = True,
                                   x_inline = False, y_inline = False,
                                   lw = 0.8, linestyle = '--', color = 'grey')
            gl.xlabel_style = {'size': 14}
            gl.ylabel_style = {'size': 14}
            gl.rotate_labels = False
        else:
            axes[n].set_xticks(LonLabel, crs = data_proj)
            axes[n].set_yticks(LatLabel, crs = data_proj)

            axes[n].set_yticklabels(LatLabel, fontsize = 14)
            axes[n].set_xticklabels(LonLabel, fontsize = 14)

            axes[n].xaxis.set_major_formatter(LonFormatter)
            axes[n].yaxis.set_major_formatter(LatFormatter)

        # Plot the data
        cs = axes[n].pcolormesh(lon, lat, data, vmin = cmins[n], vmax = cmaxs[n],
                                cmap = cmaps[n], transform = data_proj, zorder = 1)

        # Set the map extent
        axes[n].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])


        # Set the colorbar size and location
        cbax = fig.add_axes([horizontal_coords[n], 0.330, 0.34, 0.015])
        cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmaps[n], orientation = 'horizontal')
        cbar.ax.set_xlabel(color_labels[n], fontsize = 16)

        # Set the colorbar ticks
        for i in cbar.ax.xaxis.get_ticklabels():
            i.set_size(16)
        
        
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_histogram():
    '''
    Display a line histogram
    '''
    return

def make_physical_comparison_plot():
    '''
    Make plots showing calculations from true labels vs predicted labels
    '''
    return

def running_sum(data, N = 7):
    '''
    Calculate an N day (end point) running sum 
    '''
    # data = data.astype(np.float64) # Change data to float64 to prevent precision errors in the center of data

    T= data.size
    data_copy = data.copy()

    run_sum = np.zeros((T, ))
    run_sum = np.convolve(data_copy, np.ones((N,)))[(N-1):]
    # print(N, run_sum)
    return run_sum

def make_drought_fingerprint_plot(timeseries, timeseries_prediction, times, 
                                  start, variable = 'precipitation', region = 'east africa'):
    '''
    Make the drought fingerprint plot

    Fix ticks and colors
    '''
    # Get the variable short name
    sname = get_var_shortname(variable)
    
    # fig, ax = plt.subplots(figsize = [18, 8], ncols = 1)

    # # Set the title
    # fig.suptitle('Accumulated %s ending %s for %s'%(variable, start.strftime('%Y-%m-%d'), region.title()), fontsize = 18)
    # ax.plot(times, running_sum(timeseries, 300), 'b')
    # ax.set_ylabel('Precip (m)')
    # plt.savefig('test_fig.png')

    # Initialize some variables
    one_year = np.array([datetime(2000, 1, 1) + timedelta(days = t) for t in range(366)])

    months = np.array([date.month for date in times])
    days = np.array([date.day for date in times])

    # Create a set of the time series that only runs to the "start" point (start point serves as "1 day accumulation")
    ind_t = np.where(start == times)[0]
    times_plot = times[:ind_t[0]]
    data_plot = timeseries[:ind_t[0]]

    # Initialize percentiles
    percentiles_100 = []
    percentiles_98 = []
    percentiles_95 = []
    percentiles_90 = []
    percentiles_80 = []
    percentiles_70 = []
    percentiles_50 = []
    percentiles_30 = []
    percentiles_20 = []
    percentiles_10 = []
    percentiles_5 = []
    percentiles_2 = []
    percentiles_0 = []

    # Convert the time series to float64 to avoid rounding and precision errors that cause unstable summations
    timeseries = timeseries.astype(np.float64)

    # Loop backwards through time, summing each point in the data and calculation statistics relative to the N-day accumulation and day of year
    n = 1
    for t in reversed(range(times_plot.size)):
        # Get the indices for all times with the current date (up to t to avoid end of window errors)
        ind = np.where( (times_plot[t].month == months[:t]) & (times_plot[t].day == days[:t]) )[0]

        if np.invert(t == (ind_t[0]-1)):
            # Add to the data (N-day accumulation)
            data_plot[t] = data_plot[t] + data_plot[t+1]

            # Conduction a running summation of N days for statistics to compare data_plot to
            tmp = running_sum(timeseries, n)
        else:
            tmp = timeseries

        if len(ind) >= 1:
            # Calculate the maximum, minimum, median, and percentiles for the current N-day accumulation and day of year
            percentiles_100.append(np.nanmax(tmp[ind]))
            percentiles_98.append(np.nanpercentile(tmp[ind], 98))
            percentiles_95.append(np.nanpercentile(tmp[ind], 95))
            percentiles_90.append(np.nanpercentile(tmp[ind], 90))
            percentiles_80.append(np.nanpercentile(tmp[ind], 80))
            percentiles_70.append(np.nanpercentile(tmp[ind], 70))
            percentiles_50.append(np.nanmedian(tmp[ind]))
            percentiles_30.append(np.nanpercentile(tmp[ind], 30))
            percentiles_20.append(np.nanpercentile(tmp[ind], 20))
            percentiles_10.append(np.nanpercentile(tmp[ind], 10))
            percentiles_5.append(np.nanpercentile(tmp[ind], 5))
            percentiles_2.append(np.nanpercentile(tmp[ind], 2))
            percentiles_0.append(np.nanmin(tmp[ind]))
        else:
            # For the last day (when ind is empty), set the last point to NaN (last entry is needed so they have the same length as time_plot)
            percentiles_100.append(np.nan)
            percentiles_98.append(np.nan)
            percentiles_95.append(np.nan)
            percentiles_90.append(np.nan)
            percentiles_80.append(np.nan)
            percentiles_70.append(np.nan)
            percentiles_50.append(np.nan)
            percentiles_30.append(np.nan)
            percentiles_20.append(np.nan)
            percentiles_10.append(np.nan)
            percentiles_5.append(np.nan)
            percentiles_2.append(np.nan)
            percentiles_0.append(np.nan)

        n = n + 1

    # Turn the statistics into arrays and reverse them so they go forward in time 
    # Note the reversing step is not necessary, it just makes them consistent with data_plot and time_plot
    percentiles_100 = np.array(percentiles_100)[::-1]
    percentiles_98 = np.array(percentiles_98)[::-1]
    percentiles_95 = np.array(percentiles_95)[::-1]
    percentiles_90 = np.array(percentiles_90)[::-1]
    percentiles_80 = np.array(percentiles_80)[::-1]
    percentiles_70 = np.array(percentiles_70)[::-1]
    percentiles_50 = np.array(percentiles_50)[::-1]
    percentiles_30 = np.array(percentiles_30)[::-1]
    percentiles_20 = np.array(percentiles_20)[::-1]
    percentiles_10 = np.array(percentiles_10)[::-1]
    percentiles_5 = np.array(percentiles_5)[::-1]
    percentiles_2 = np.array(percentiles_2)[::-1]
    percentiles_0 = np.array(percentiles_0)[::-1]

    # Calculate accumulations for the predictions
    prediction_plot = timeseries_prediction
    for t in reversed(range(prediction_plot.size)):
        if t == (prediction_plot.size - 1):
            continue
        else:
            prediction_plot[t] = prediction_plot[t] + prediction_plot[t+1]

    # Information for colors
    colors = {
        'minimum': 'white', 'W0': '#00FFFF', 'W1': '#ADD8E6', 'W2': '#7BC8F6', 'W3': '#069AF3', 'W4': '#0000FF',
        'median': 'black',
        'D0': '#FFFF14', 'D1': '#FCD37F', 'D2': '#FFAA00', 'D3': '#E60000', 'D4': '#730000', 'maximum': 'white'
    }

    ncols = 2

    # Functions to convert between the primary and secondary x-axes
    def accum_days_to_datetimes(x, initial_day):
        return -1*(x + mdates.date2num(initial_day))
        # return mdates.num2date(mdates.date2num(start) - x)
    
    def datetimes_to_accum_days(y, initial_day):
        return -1*(y - mdates.date2num(initial_day))
        # return mdates.date2num(start) - mdates.date2num(x)

    # Create the plot
    fig, ax = plt.subplots(figsize = [15, 10], ncols = ncols)

    # Initialize some information
    titles = ['S2S Range', 'Short Range']
    ndays = [90, 180]

    formatters = [
        DateFormatter('%b'),
        DateFormatter('%Y')
    ]
    ticks_short = []# datetime(start.year, (start.month - m)%12, 15) for m in range(int(times_plot.size/30))] # Make ticks for short range; one tick per month
    for m in range(int(times_plot.size/30)):
        if (start.month - m) <= 0:
            ticks_short.append(datetime(start.year, ((start.month - m)%12) + 1, 15))
        else:
            ticks_short.append(datetime(start.year, start.month - m, 15))
            
    ticks_long = [datetime(start.year - m, 6, 1) for m in range(int(times_plot.size/365))] # Make ticks for long range; one tick per year 
    primary_ticks = [ticks_short, ticks_long]
    secondary_ticks = [np.arange(ndays[0]+1)[::30], np.arange(ndays[1]+1)[::30]] # Ticks for secondary axis

    # Set the title
    fig.suptitle('Accumulated %s ending %s for %s'%(variable, start.strftime('%Y-%m-%d'), region.title()), y = 0.95, fontsize = 18)

    for m in range(ncols):
        # Set short and long range titles
        ax[m].set_title(titles[m], fontsize = 18)
                
        # Plot the statistics
        ax[m].plot(times_plot[::-1], percentiles_100[::-1], color = 'white', label = 'maximum')
        ax[m].fill_between(times_plot[::-1], percentiles_98[::-1], np.nanmax(percentiles_100), color = colors['W4'], alpha = 0.9, label = 'W4')
        ax[m].fill_between(times_plot[::-1], percentiles_95[::-1], percentiles_98[::-1], color = colors['W3'], alpha = 0.9, label = 'W3')
        ax[m].fill_between(times_plot[::-1], percentiles_90[::-1], percentiles_95[::-1], color = colors['W2'], alpha = 0.9, label = 'W2')
        ax[m].fill_between(times_plot[::-1], percentiles_80[::-1], percentiles_90[::-1], color = colors['W1'], alpha = 0.9, label = 'W1')
        ax[m].fill_between(times_plot[::-1], percentiles_70[::-1], percentiles_80[::-1], color = colors['W0'], alpha = 0.9, label = 'W0')
        ax[m].plot(times_plot[::-1], percentiles_50[::-1], color = 'black', label = 'median')
        ax[m].fill_between(times_plot[::-1], percentiles_20[::-1], percentiles_30[::-1], color = colors['D0'], alpha = 0.9, label = 'D0')
        ax[m].fill_between(times_plot[::-1], percentiles_10[::-1], percentiles_20[::-1], color = colors['D1'], alpha = 0.9, label = 'D1')
        ax[m].fill_between(times_plot[::-1], percentiles_5[::-1], percentiles_10[::-1], color = colors['D2'], alpha = 0.9, label = 'D2')
        ax[m].fill_between(times_plot[::-1], percentiles_2[::-1], percentiles_5[::-1], color = colors['D3'], alpha = 0.9, label = 'D3')
        ax[m].fill_between(times_plot[::-1], 0, percentiles_2[::-1], color = colors['D4'], alpha = 0.9, label = 'D4')
        ax[m].plot(times_plot[::-1], percentiles_0[::-1], color = 'white', label = 'minimum')

        # Plot the actual data
        ax[m].plot(times_plot[::-1], data_plot[::-1], color = 'black', linewidth = 3.0, label = 'True Labels')

        # Plot the predictions of the variable for the S2S plot only (since predictions are only 90 days out)
        if 'S2S' in titles[m]:
            times_prediction = times_plot[-prediction_plot.size:]
            ax[m].plot(times_prediction[::-1], prediction_plot[::-1], color = 'black', linewidth = 3.0, linestyle = '--', label = 'CrossFormer')

        # Make a secondary axis to show days of accumulated variable
        sec_xaxis = ax[m].secondary_xaxis(location = -0.07, 
                                          functions = (partial(datetimes_to_accum_days, initial_day = start), 
                                                       partial(accum_days_to_datetimes, initial_day = start)))

        sec_xaxis.xaxis.set_ticks(secondary_ticks[m])#, labels = secondary_ticks[m].astype(str).tolist())

        #sec_xaxis.grid(True, axis = 'x', color = 'black', linewidth = 2, linestyle = ':')
        # A work around secondary_xaxis.grid() plotting nothing; manually make a line for each tick on the main axis
        for line in sec_xaxis.get_xticks():
            if (line == 0) | (line == ndays[0]) | (line == ndays[1]):
                ls = '-'
            else:
                ls = ':'

            date = start - timedelta(days = line.astype(float))
            ax[m].axvline(date, color = 'k', linewidth = 2, linestyle = ls, zorder = 1)
        
        # Make the ticks large enough to connect the secondary axis to the graph
        sec_xaxis.tick_params(axis = 'x', direction = 'in', size = 40, width = 2)#, linestyle = '-')

        # Make the legend
        ax[m].legend(loc = 'upper left')

        # Set the x ticks
        ax[m].set_xticks(ticks_short)

        # Set x and y limits
        ax[m].set_xlim([start, start - timedelta(days = ndays[m])])
        ax[m].set_ylim([0, 1.05*np.nanmax(percentiles_100[-ndays[m]:])])

        ax[m].xaxis.set_major_formatter(DateFormatter('%b'))

        # Set x and y labels
        sec_xaxis.set_xlabel('Days of Accumulated %s'%variable, fontsize = 18)
        ax[m].set_ylabel('%s (%s)'%(variable.title(), units[sname]), fontsize = 18)

        # Set tick sizes
        for i in ax[m].xaxis.get_ticklabels() + ax[m].yaxis.get_ticklabels() + sec_xaxis.xaxis.get_ticklabels():
            i.set_size(18)

    # Save figure
    savename = 'fingerprint_plot_for_%s_%s.png'%(variable, region)
    plt.savefig('%s'%(savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_fingerprint_plot(timeseries, times, start, variable = 'precipitation', region = 'east africa'):
    '''
    Make the drought fingerprint plot

    Fix ticks and colors
    '''
    # Get the variable short name
    sname = get_var_shortname(variable)
    
    # fig, ax = plt.subplots(figsize = [18, 8], ncols = 1)

    # # Set the title
    # fig.suptitle('Accumulated %s ending %s for %s'%(variable, start.strftime('%Y-%m-%d'), region.title()), fontsize = 18)
    # ax.plot(times, running_sum(timeseries, 300), 'b')
    # ax.set_ylabel('Precip (m)')
    # plt.savefig('test_fig.png')

    # Initialize some variables
    one_year = np.array([datetime(2000, 1, 1) + timedelta(days = t) for t in range(366)])

    months = np.array([date.month for date in times])
    days = np.array([date.day for date in times])

    # Create a set of the time series that only runs to the "start" point (start point serves as "1 day accumulation")
    ind_t = np.where(start == times)[0]
    times_plot = times[:ind_t[0]]
    data_plot = timeseries[:ind_t[0]]

    # Initialize percentiles
    percentiles_100 = []
    percentiles_98 = []
    percentiles_95 = []
    percentiles_90 = []
    percentiles_80 = []
    percentiles_70 = []
    percentiles_50 = []
    percentiles_30 = []
    percentiles_20 = []
    percentiles_10 = []
    percentiles_5 = []
    percentiles_2 = []
    percentiles_0 = []

    # Convert the time series to float64 to avoid rounding and precision errors that cause unstable summations
    timeseries = timeseries.astype(np.float64)

    # Loop backwards through time, summing each point in the data and calculation statistics relative to the N-day accumulation and day of year
    n = 1
    for t in reversed(range(times_plot.size)):
        # Get the indices for all times with the current date (up to t to avoid end of window errors)
        ind = np.where( (times_plot[t].month == months[:t]) & (times_plot[t].day == days[:t]) )[0]

        if np.invert(t == (ind_t[0]-1)):
            # Add to the data (N-day accumulation)
            data_plot[t] = data_plot[t] + data_plot[t+1]

            # Conduction a running summation of N days for statistics to compare data_plot to
            tmp = running_sum(timeseries, n)
        else:
            tmp = timeseries

        if len(ind) >= 1:
            # Calculate the maximum, minimum, median, and percentiles for the current N-day accumulation and day of year
            percentiles_100.append(np.nanmax(tmp[ind]))
            percentiles_98.append(np.nanpercentile(tmp[ind], 98))
            percentiles_95.append(np.nanpercentile(tmp[ind], 95))
            percentiles_90.append(np.nanpercentile(tmp[ind], 90))
            percentiles_80.append(np.nanpercentile(tmp[ind], 80))
            percentiles_70.append(np.nanpercentile(tmp[ind], 70))
            percentiles_50.append(np.nanmedian(tmp[ind]))
            percentiles_30.append(np.nanpercentile(tmp[ind], 30))
            percentiles_20.append(np.nanpercentile(tmp[ind], 20))
            percentiles_10.append(np.nanpercentile(tmp[ind], 10))
            percentiles_5.append(np.nanpercentile(tmp[ind], 5))
            percentiles_2.append(np.nanpercentile(tmp[ind], 2))
            percentiles_0.append(np.nanmin(tmp[ind]))
        else:
            # For the last day (when ind is empty), set the last point to NaN (last entry is needed so they have the same length as time_plot)
            percentiles_100.append(np.nan)
            percentiles_98.append(np.nan)
            percentiles_95.append(np.nan)
            percentiles_90.append(np.nan)
            percentiles_80.append(np.nan)
            percentiles_70.append(np.nan)
            percentiles_50.append(np.nan)
            percentiles_30.append(np.nan)
            percentiles_20.append(np.nan)
            percentiles_10.append(np.nan)
            percentiles_5.append(np.nan)
            percentiles_2.append(np.nan)
            percentiles_0.append(np.nan)

        n = n + 1

    # Turn the statistics into arrays and reverse them so they go forward in time 
    # Note the reversing step is not necessary, it just makes them consistent with data_plot and time_plot
    percentiles_100 = np.array(percentiles_100)[::-1]
    percentiles_98 = np.array(percentiles_98)[::-1]
    percentiles_95 = np.array(percentiles_95)[::-1]
    percentiles_90 = np.array(percentiles_90)[::-1]
    percentiles_80 = np.array(percentiles_80)[::-1]
    percentiles_70 = np.array(percentiles_70)[::-1]
    percentiles_50 = np.array(percentiles_50)[::-1]
    percentiles_30 = np.array(percentiles_30)[::-1]
    percentiles_20 = np.array(percentiles_20)[::-1]
    percentiles_10 = np.array(percentiles_10)[::-1]
    percentiles_5 = np.array(percentiles_5)[::-1]
    percentiles_2 = np.array(percentiles_2)[::-1]
    percentiles_0 = np.array(percentiles_0)[::-1]

    # Information for colors
    colors = {
        'minimum': 'white', 'W0': '#00FFFF', 'W1': '#ADD8E6', 'W2': '#7BC8F6', 'W3': '#069AF3', 'W4': '#0000FF',
        'median': 'black',
        'D0': '#FFFF14', 'D1': '#FCD37F', 'D2': '#FFAA00', 'D3': '#E60000', 'D4': '#730000', 'maximum': 'white'
    }

    ncols = 2

    # Functions to convert between the primary and secondary x-axes
    def accum_days_to_datetimes(x, initial_day):
        return -1*(x + mdates.date2num(initial_day))
        # return mdates.num2date(mdates.date2num(start) - x)
    
    def datetimes_to_accum_days(y, initial_day):
        return -1*(y - mdates.date2num(initial_day))
        # return mdates.date2num(start) - mdates.date2num(x)

    # Create the plot
    fig, ax = plt.subplots(figsize = [15, 10], ncols = ncols)

    # Initialize some information
    titles = ['Short Range', 'Long Range']
    ndays = [180, 1825]

    formatters = [
        DateFormatter('%b'),
        DateFormatter('%Y')
    ]
    ticks_short = []# datetime(start.year, (start.month - m)%12, 15) for m in range(int(times_plot.size/30))] # Make ticks for short range; one tick per month
    for m in range(int(times_plot.size/30)):
        if (start.month - m) <= 0:
            ticks_short.append(datetime(start.year, ((start.month - m)%12) + 1, 15))
        else:
            ticks_short.append(datetime(start.year, start.month - m, 15))
            
    ticks_long = [datetime(start.year - m, 6, 1) for m in range(int(times_plot.size/365))] # Make ticks for long range; one tick per year 
    primary_ticks = [ticks_short, ticks_long]
    secondary_ticks = [np.arange(ndays[0]+1)[::30], np.arange(ndays[1]+1)[::365]] # Ticks for secondary axis

    # Set the title
    fig.suptitle('Accumulated %s ending %s for %s'%(variable, start.strftime('%Y-%m-%d'), region.title()), y = 0.95, fontsize = 18)

    for m in range(ncols):
        # Set short and long range titles
        ax[m].set_title(titles[m], fontsize = 18)
                
        # Plot the statistics
        ax[m].plot(times_plot[::-1], percentiles_100[::-1], color = 'white', label = 'maximum')
        ax[m].fill_between(times_plot[::-1], percentiles_98[::-1], np.nanmax(percentiles_100), color = colors['W4'], alpha = 0.9, label = 'W4')
        ax[m].fill_between(times_plot[::-1], percentiles_95[::-1], percentiles_98[::-1], color = colors['W3'], alpha = 0.9, label = 'W3')
        ax[m].fill_between(times_plot[::-1], percentiles_90[::-1], percentiles_95[::-1], color = colors['W2'], alpha = 0.9, label = 'W2')
        ax[m].fill_between(times_plot[::-1], percentiles_80[::-1], percentiles_90[::-1], color = colors['W1'], alpha = 0.9, label = 'W1')
        ax[m].fill_between(times_plot[::-1], percentiles_70[::-1], percentiles_80[::-1], color = colors['W0'], alpha = 0.9, label = 'W0')
        ax[m].plot(times_plot[::-1], percentiles_50[::-1], color = 'black', label = 'median')
        ax[m].fill_between(times_plot[::-1], percentiles_20[::-1], percentiles_30[::-1], color = colors['D0'], alpha = 0.9, label = 'D0')
        ax[m].fill_between(times_plot[::-1], percentiles_10[::-1], percentiles_20[::-1], color = colors['D1'], alpha = 0.9, label = 'D1')
        ax[m].fill_between(times_plot[::-1], percentiles_5[::-1], percentiles_10[::-1], color = colors['D2'], alpha = 0.9, label = 'D2')
        ax[m].fill_between(times_plot[::-1], percentiles_2[::-1], percentiles_5[::-1], color = colors['D3'], alpha = 0.9, label = 'D3')
        ax[m].fill_between(times_plot[::-1], 0, percentiles_2[::-1], color = colors['D4'], alpha = 0.9, label = 'D4')
        ax[m].plot(times_plot[::-1], percentiles_0[::-1], color = 'white', label = 'minimum')

        # Plot the actual data
        ax[m].plot(times_plot[::-1], data_plot[::-1], color = 'black', linewidth = 3.0, label = 'observations')

        # Make a secondary axis to show days of accumulated variable
        sec_xaxis = ax[m].secondary_xaxis(location = -0.07, 
                                          functions = (partial(datetimes_to_accum_days, initial_day = start), 
                                                       partial(accum_days_to_datetimes, initial_day = start)))

        sec_xaxis.xaxis.set_ticks(secondary_ticks[m])#, labels = secondary_ticks[m].astype(str).tolist())

        #sec_xaxis.grid(True, axis = 'x', color = 'black', linewidth = 2, linestyle = ':')
        # A work around secondary_xaxis.grid() plotting nothing; manually make a line for each tick on the main axis
        for line in sec_xaxis.get_xticks():
            if (line == 0) | (line == ndays[0]) | (line == ndays[1]):
                ls = '-'
            else:
                ls = ':'

            date = start - timedelta(days = line.astype(float))
            ax[m].axvline(date, color = 'k', linewidth = 2, linestyle = ls, zorder = 1)
        
        # Make the ticks large enough to connect the secondary axis to the graph
        sec_xaxis.tick_params(axis = 'x', direction = 'in', size = 40, width = 2)#, linestyle = '-')

        # Make the legend
        ax[m].legend(loc = 'upper left')

        # Set the x ticks
        ax[m].set_xticks(primary_ticks[m])

        # Set x and y limits
        ax[m].set_xlim([start, start - timedelta(days = ndays[m])])
        ax[m].set_ylim([0, 1.05*np.nanmax(percentiles_100[-ndays[m]:])])

        ax[m].xaxis.set_major_formatter(formatters[m])

        # Set x and y labels
        sec_xaxis.set_xlabel('Days of Accumulated %s'%variable, fontsize = 18)
        ax[m].set_ylabel('%s (%s)'%(variable.title(), units[sname]), fontsize = 18)

        # Set tick sizes
        for i in ax[m].xaxis.get_ticklabels() + ax[m].yaxis.get_ticklabels() + sec_xaxis.xaxis.get_ticklabels():
            i.set_size(18)

    # Save figure
    savename = 'drought_fingerprint_plot_for_%s_%s.png'%(variable, region)
    plt.savefig('%s'%(savename), bbox_inches = 'tight')
    plt.show(block = False)


if __name__ == '__main__':
    # Currently just a test for the drought plot
    import zarr
    import numpy as np
    from datetime import datetime, timedelta
    from netCDF4 import Dataset
    # from plotting import make_drought_fingerprint_plot

    variable = []
    # Southwestern US
    lower_lat = 30
    upper_lat = 40
    lower_lon = 240
    upper_lon = 255
    # Southern US
    # lower_lat = 29
    # upper_lat = 36
    # lower_lon = 258
    # upper_lon = 274
    # Central US
    # lower_lat = 35
    # upper_lat = 42
    # lower_lon = 265
    # upper_lon = 275
    # Eastern Africa
    # lower_lat = -5
    # upper_lat = 20
    # lower_lon = 35
    # upper_lon = 52

    years = np.arange(2000, 2024)

    for year in years:
        print(year)
        root = zarr.open_group('surface.%d.zarr'%year)
        # root = zarr.open_group('diagnostic.%d.zarr'%year)
        # print(np.nanmin(root['swvl1'][:]), np.nanmax(root['swvl1'][:]), np.nanmean(root['swvl1'][:]))
        lon_ind = np.where( (root['longitude'][0,:] >= lower_lon) & (root['longitude'][0,:] <= upper_lon) )[0]
        lat_ind = np.where( (root['latitude'][:,0] >= lower_lat) & (root['latitude'][:,0] <= upper_lat) )[0]
        tmp = np.nanmean(root['tp'][:,:,lon_ind], axis = -1)
        # tmp = np.nanmean(np.abs(root['swvl1'][:,:,lon_ind]), axis = -1)
        variable.append(np.nanmean(tmp[:,lat_ind], axis = -1))

    variable = np.concatenate(variable)

    start = datetime(2000, 1, 1)
    times = np.array([start + timedelta(days = t) for t in range(variable.size)])

    start = datetime(2018, 9, 1)

    start_str = (start - timedelta(days = 90)).isoformat()
    hours = np.arange(24, 2160+1, 24)
    # Load in predictions
    prediction = []
    for hour in hours:
        print('Forecast %d day'%int(hour/24))
        with Dataset('/scratch/rarrell/credit_model/forecasts/%sZ/pred_%sZ_%03d.nc'%(start_str[:-6], start_str[:-6], hour), 'r') as nc:
            tmp = np.nanmean(nc.variables['tp'][:])
            # tmp = np.nanmean(nc.variables['swvl1'][:])
            prediction.append(tmp)
    prediction = np.array(prediction)

    make_drought_fingerprint_plot(variable, prediction, times, start, variable = 'precipitation', region = 'Southwestern US')
