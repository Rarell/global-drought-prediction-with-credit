import numpy as np
from scipy import fft
from datetime import datetime, timedelta
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import colorbar as mcolorbar
from netCDF4 import Dataset
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from itertools import product
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from path_to_raw_datasets import get_var_shortname
from metric_calculations import calculate_acc_in_space, calculate_rmse_in_space
from utils import least_squares, new_sort

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
    'evi': 'EVI',
    'ndvi': 'NDVI',
    'lai': 'LAI',
    'fpar': 'fPAR',
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
    '200 mb u': '200 mb U Wind', '500 mb u': '500 mb U Wind',
    '200 mb v': '200 mb V Wind', '500 mb v': '500 mb V Wind',
    '200 mb z': '200 mb Geopotential', '500 mb z': '500 mb Geopotential',
    '200 mb q_tot': '200 mb Total Specific Humidity', '500 mb q_tot': '500 mb Total Specific Humidity',
    'tair': '2 Metre Temperature',
    'sp': 'Surface Pressure',
    'd2m': '2 Metre Dewpoint Temperature',
    'tp': '1 Day Accumulated Precipitation', 
    'tp_7d': '7 Day Accumulated Precipitation', 
    'tp_14d': '14 Day Accumulated Precipitation',
    'tp_30d': '30 Day Accumulated Precipitation',
    'e': '1 Day Accumulated Evaporation',
    'pev': '1 Day Accumulated Potential Evaporation',
    'evi': 'EVI', 'ndvi': 'NDVI', 'lai': 'LAI', 'fpar': 'fPAR',
    'cvh': 'High Vegetation Coverage',
    'cvl': 'Low Vegetation Coverage',
    'ssr': 'Net Surface Shortwave Radiation',
    'ws': '2 Metre Wind Speed',
    'fg10': '10 Metre Wind Gusts',
    'swvl1': '0 - 10 cm Soil Moisture', 
    'swvl2': '10 - 40 cm Soil Moisture', 
    'swvl3': '40 - 100 cm Soil Moisture', 
    'swvl4': '100 - 200 cm Soil Moisture',
    'fdii1': '0 - 10 cm FDII', 
    'fdii2': '10 - 40 cm FDII', 
    'fdii3': '40 - 100 cm FDII', 
    'fdii4': '100 - 200 cm FDII',
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
    'tp': 'mm', 'tp_7d': 'mm', 'tp_14d': 'mm', 'tp_30d': 'mm',
    'e': 'mm',
    'pev': 'mm',
    'evi': 'unitless', 'ndvi': 'unitless', 'lai': 'unitless', 'fpar': 'unitless',
    'cvh': 'unitless', 'cvl': 'unitless',
    'ssr': r'J m$^{-2}$',
    'ws': r'm s$^{-1}$',
    'fg10': r'm s$^{-1}$',
    'swvl1': r'kg m$^{-2}$', 'swvl2': r'kg m$^{-2}$', 'swvl3': r'kg m$^{-2}$', 'swvl4': r'kg m$^{-2}$',
    'fdii1': 'unitless', 'fdii2': 'unitless', 'fdii3': 'unitless', 'fdii4': 'unitless',
    'sesr': 'unitless'
}
color_information = {
    '200 mb u': {'climits': [-60, 60], 'climits_metric': [3, 20], 'hist_values': [-25, 65.5], 'max_counts': 350, 'cname': 'PuOr'}, 
    '500 mb u': {'climits': [-40, 40], 'climits_metric': [3, 15], 'hist_values': [-20, 40.5], 'max_counts': 300, 'cname': 'PuOr'},
    '200 mb v': {'climits': [-35, 35], 'climits_metric': [3, 25], 'hist_values': [-30, 35.5], 'max_counts': 800, 'cname': 'PuOr'}, 
    '500 mb v': {'climits': [-30, 30], 'climits_metric': [3, 15], 'hist_values': [-30, 30.5], 'max_counts': 800, 'cname': 'PuOr'},
    '200 mb z': {'climits': [110000, 125000], 'climits_metric': [0, 3000], 'hist_values': [112000, 124500], 'max_counts': 800, 'cname': 'RdBu_r'}, 
    '500 mb z': {'climits': [49000, 59000], 'climits_metric': [0, 2000], 'hist_values': [51000, 58500], 'max_counts': 800, 'cname': 'RdBu_r'},
    '200 mb q_tot': {'climits': [0, 1.5e-4], 'climits_metric': [1e-5, 8e-5], 'hist_values': [0, 0.00055], 'max_counts': 800, 'cname': 'BrBG'}, 
    '500 mb q_tot': {'climits': [0, 0.004], 'climits_metric': [0.0005, 0.0020], 'hist_values': [0, 0.0085], 'max_counts': 600, 'cname': 'BrBG'},
    'tair': {'climits': [210, 330], 'climits_metric': [0.5, 5.5], 'hist_values': [260, 315], 'max_counts': 300, 'cname': 'RdBu_r'},
    'sp': {'climits': [80000, 105000], 'climits_metric': [1000, 4000], 'hist_values': [70000, 105900], 'max_counts': 700, 'cname': 'RdBu_r'},
    'd2m':  {'climits': [200, 320], 'climits_metric': [0.5, 9.5], 'hist_values': [250, 305], 'max_counts': 300, 'cname': 'BrBG'}, # Fine tune
    'tp': {'climits': [0, 10.0], 'climits_metric': [2.0, 25.0], 'hist_values': [0, 105.0], 'max_counts': 1000, 'cname': 'Greens'}, # Max color count, if used, is around 600 - 700
    'tp_7d': {'climits': [0, 10.0], 'climits_metric': [2.0, 25.0], 'hist_values': [0, 350.0], 'max_counts': 1000, 'cname': 'Greens'}, # Fine tune
    'tp_14d': {'climits': [0, 200.0], 'climits_metric': [2.0, 25.0], 'hist_values': [0, 750.0], 'max_counts': 1000, 'cname': 'Greens'}, # Fine tune
    'tp_30d': {'climits': [0, 500.0], 'climits_metric': [2.0, 100.0], 'hist_values': [0, 1250.0], 'max_counts': 1000, 'cname': 'Greens'},
    'e': {'climits': [-5.0, 5.00], 'climits_metric': [0.1, 5.0], 'hist_values': [-6.0, 7.0], 'max_counts': 500, 'cname': 'BrBG'}, 
    'pev': {'climits': [0, 15.0], 'climits_metric': [0.5, 6.0], 'hist_values': [0, 18.0], 'max_counts': 500, 'cname': 'BrBG_r'}, 
    'evi': {'climits': [0, 0.6], 'climits_metric': [0.01, 0.4], 'hist_values': [0, 0.6], 'max_counts': 100, 'cname': 'BrBG'}, ### FINE TUNE
    'ndvi': {'climits': [0, 1.0], 'climits_metric': [0.01, 0.4], 'hist_values': [0, 1.0], 'max_counts': 100, 'cname': 'BrBG'}, ### FINE TUNE
    'fpar': {'climits': [0, 1.0], 'climits_metric': [0.01, 0.4], 'hist_values': [0, 0.9], 'max_counts': 100, 'cname': 'BrBG'}, ### FINE TUNE
    'lai': {'climits': [0, 4.5], 'climits_metric': [0.0001, 0.4], 'hist_values': [0, 4.5], 'max_counts': 100, 'cname': 'BrBG'}, ### FINE TUNE  
    'cvh': {'climits': [0, 1.0], 'climits_metric': [0.030, 0.40], 'cname': 'BrBG'},
    'cvl': {'climits': [0, 1.0], 'climits_metric': [0.030, 0.30], 'cname': 'BrBG'}, 
    'ssr': {'climits': [0, 30000000], 'cname': 'Reds'}, # Fine tune
    'ws':  {'climits': [0, 20], 'cname': 'PuOr'}, # Fine tune
    'fg10': {'climits': [0, 25], 'cname': 'PuOr'}, # Fine tune
    'swvl1': {'climits': [0, 55], 'climits_metric': [2.1, 6.65], 'hist_values': [0, 45.5], 'max_counts': 300, 'cname': 'BrBG'}, # Max color count, if used, is around 25
    'swvl2': {'climits': [0, 55], 'climits_metric': [2.1, 13.5], 'hist_values': [0, 45.5], 'max_counts': 300, 'cname': 'BrBG'},
    'swvl3': {'climits': [0, 55], 'climits_metric': [2.1, 18.0], 'hist_values': [0, 45.5], 'max_counts': 300, 'cname': 'BrBG'},
    'swvl4': {'climits': [0, 55], 'climits_metric': [2.1, 24.0], 'hist_values': [0, 45.5], 'max_counts': 300, 'cname': 'BrBG'},
    'fdii1': {'climits': [0, 50], 'climits_metric': [0.1, 10.0], 'hist_values': [0, 20.5], 'max_counts': 1000, 'cname': 'Spectral_r'},
    'fdii2': {'climits': [0, 50], 'climits_metric': [0.1, 10.0], 'hist_values': [0, 20.5], 'max_counts': 1000, 'cname': 'Spectral_r'},
    'fdii3': {'climits': [0, 50], 'climits_metric': [0.1, 8.0], 'hist_values': [0, 20.5], 'max_counts': 1000, 'cname': 'Spectral_r'},
    'fdii4': {'climits': [0, 50], 'climits_metric': [0.1, 8.0], 'hist_values': [0, 20.5], 'max_counts': 1000, 'cname': 'Spectral_r'},
    'sesr': {'climits': [-3.0, 3.0], 'climits_metric': [0.0, 3.0], 'hist_values': [-3, 3], 'max_counts': 500, 'cname': 'BrBG'} 
}
subset_information = {
    'africa': {'map_extent': [-35, 35, 335, 53], 'wspace': 0.18, 'hspace': -0.40, 'hspace_anomaly': 0.15, 'wspace_metric': 0.17,
            #    'colorbar_coord': [[0.925, 0.535, 0.020, 0.21], [0.925, 0.235, 0.020, 0.21]],
               'colorbar_coord': [[0.105, 0.355, 0.60, 0.020], [0.725, 0.355, 0.185, 0.020]], 
            #    'colorbar_coord_anomaly': [[0.925, 0.565, 0.020, 0.26], [0.925, 0.165, 0.020, 0.26]], 
               'colorbar_coord_anomaly': [[0.105, 0.325, 0.53, 0.020], [0.670, 0.325, 0.23, 0.020]], 
               'colorbar_coord_metric': [0.330, 0.34, 0.015]},
    'conus': {'map_extent': [24, 55, 230, 300], 'wspace': 0.18, 'hspace': -0.70, 'hspace_anomaly': -0.60, 'wspace_metric': 0.17,
              'colorbar_coord': [[0.925, 0.525, 0.020, 0.12], [0.925, 0.345, 0.020, 0.12]], 
              'colorbar_coord_anomaly': [[0.925, 0.515, 0.020, 0.175], [0.925, 0.295, 0.020, 0.175]], 
              'colorbar_coord_metric': [0.395, 0.34, 0.015]},
    'south central': {'map_extent': [29, 36, 258, 274], 'wspace': 0.18, 'wspace_metric': 0.17}
}

anomaly_plotting_information = {
    'tair':  {'climits': [-15, 15], 'difference_climits': [-10, 10], 'color_true': '#8C000F', 'color_pred': '#DC143C'},
    'd2m':  {'climits': [-15, 15], 'difference_climits': [-15, 15], 'color_true': '#054907', 'color_pred': '#15B01A'}, # Fine tune
    'tp': {'climits': [-30, 30], 'difference_climits': [-50, 50], 'color_true': '#054907', 'color_pred': '#15B01A'}, # Max color count, if used, is around 600 - 700
    'tp_30d': {'climits': [-100, 100], 'difference_climits': [-200, 200], 'color_true': '#054907', 'color_pred': '#15B01A'},
    'e': {'climits': [-4.0, 4.0], 'difference_climits': [-4.0, 4.0], 'color_true': '#00008B', 'color_pred': '#069AF3'}, 
    'pev': {'climits': [-10.0, 10.0], 'difference_climits': [-8.0, 8.0], 'color_true': '#00008B', 'color_pred': '#069AF3'}, 
    'evi': {'climits': [-0.15, 0.15], 'difference_climits': [-0.15, 0.15], 'color_true': '#054907', 'color_pred': '#15B01A'}, ### FINE TUNE
    'ndvi': {'climits': [-0.3, 0.3], 'difference_climits': [-0.4, 0.4], 'color_true': '#054907', 'color_pred': '#15B01A'}, ### FINE TUNE
    'fpar': {'climits': [-0.3, 0.3], 'difference_climits': [-0.4, 0.4], 'color_true': '#054907', 'color_pred': '#15B01A'}, ### FINE TUNE
    'lai': {'climits': [-2.0, 2.0], 'difference_climits': [-2.5, 2.5], 'color_true': '#054907', 'color_pred': '#15B01A'}, ### FINE TUNE  
    'swvl1': {'climits': [-20, 20], 'difference_climits': [-10, 10], 'color_true': '#00008B', 'color_pred': '#069AF3'}, # Max color count, if used, is around 25
    'swvl2': {'climits': [-40, 40], 'difference_climits': [-45, 45], 'color_true': '#00008B', 'color_pred': '#069AF3'},
    'swvl3': {'climits': [-40, 40], 'difference_climits': [-45, 45], 'color_true': '#00008B', 'color_pred': '#069AF3'},
    'swvl4': {'climits': [-40, 40], 'difference_climits': [-45, 45], 'color_true': '#00008B', 'color_pred': '#069AF3'},
    'fdii1': {'climits': [-20, 20], 'difference_climits': [-20, 20]},
    'fdii2': {'climits': [-20, 20], 'difference_climits': [-20, 20]},
    'fdii3': {'climits': [-30, 30], 'difference_climits': [-30, 30]},
    'fdii4': {'climits': [-30, 30], 'difference_climits': [-30, 30]},
    'sesr': {'climits': [-3.0, 3.0], 'difference_climits': [-3.0, 3.0]} 
}


# Function to generate a plot of performance metrics with forecast hour
def plot_metric(metric, y, metric_name, var_name, climatology = None, persistence = None,
                month = 6, year = 2018, add_variation = False,
                path = './Figures/', savename = 'timeseries.png'):
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

    # Make the average plot
    if len(metric.shape) > 1:
        metric_average = np.nanmean(metric, axis = 0)
    else:
        metric_average = metric

    if add_variation:
        metric_std = np.nanstd(metric, axis = 0)

    # Make the plot
    fig, ax = plt.subplots(figsize = [12, 8])

    # Set the title
    if (month is None) & (year is None):
        # ax.set_title('Average %s Forecast Skill vs Forecast Length\nfor All Forecasts'%(variable), fontsize = 18)
        ax.set_title('%s'%(variable), fontsize = 18)
    elif (year is None) & ((month >= 3) & (month <= 8)):
        ax.set_title('%s'%(variable), fontsize = 18)
        # ax.set_title('Average %s Forecast Skill vs Forecast Length\nfor All Spring and Summer Forecasts'%(variable), fontsize = 18)
    elif (year is None) & ((month >= 9) | (month <= 2)):
        ax.set_title('%s'%(variable), fontsize = 18)
        # ax.set_title('Average %s Forecast Skill vs Forecast Length\nfor All Fall and Winter Forecasts'%(variable), fontsize = 18)
    else:
        ax.set_title('%s Forecast Skill vs Forecast Length\nfor Forecasts starting in %d, %d'%(variable, month, year), fontsize = 18)
    
    # Make the plot
    if (len(metric.shape) > 1) & (add_variation == False):
        I, J = metric.shape

        for i in range(I):
            ax.plot(y, metric[i,:], color = 'grey', linewidth = 0.5)

    # Plot Average line
    ax.plot(y, metric_average, color = 'k', linewidth = 2.5, label = 'CrossFormer')
    if add_variation:
        ax.fill_between(y, metric_average, metric_average + metric_std, color = 'grey', alpha = 0.5)
        ax.fill_between(y, metric_average, metric_average - metric_std, color = 'grey', alpha = 0.5)

    # Plot climatology and persistence (note these include forecasts at day = 0, so start from 1)
    if climatology is not None:
        ax.plot(y, np.nanmean(climatology, axis = 0), color = 'b', linewidth = 2.5, label = 'Climatology')
        # if add_variation:
            # ax.fill_between(y, np.nanmean(climatology, axis = 0), 
            #                 np.nanmean(climatology, axis = 0) + np.nanstd(climatology, axis = 0), 
            #                 color = 'g', alpha = 0.5)
            # ax.fill_between(y, np.nanmean(climatology, axis = 0), 
            #                 np.nanmean(climatology, axis = 0) - np.nanstd(climatology, axis = 0), 
            #                 color = 'g', alpha = 0.5)

    if persistence is not None:
        ax.plot(y, np.nanmean(persistence, axis = 0), color = 'r', linewidth = 2.5, label = 'Persistence')
        # if add_variation:
        #     ax.fill_between(y, np.nanmean(persistence, axis = 0), 
        #                     np.nanmean(persistence, axis = 0) + np.nanstd(persistence, axis = 0), 
        #                     color = 'b', alpha = 0.5)
        #     ax.fill_between(y, np.nanmean(persistence, axis = 0), 
        #                     np.nanmean(persistence, axis = 0) - np.nanstd(persistence, axis = 0), 
        #                     color = 'b', alpha = 0.5)

    if (climatology is not None) | (persistence is not None):
        ax.legend(fontsize = 18)
        
    # Add labels
    if metric_name.upper() == 'ACC':
        ax.set_ylabel(metric_name.upper(), fontsize = 18)
    else:
        ax.set_ylabel('%s [%s]'%(metric_name.upper(), units[var_name]), fontsize = 18)
    ax.set_xlabel('Lead Time [Days]', fontsize = 18)
    # ax.set_xlabel('Forecast Day', fontsize = 18)

    # Tick information
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(18)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


# Function to generate a plot of performance metrics with forecast hour
def plot_violin(true, pred, var_name, path = './Figures/', savename = 'violin.png'):
    '''
    Create violin plot of metrics at different lead times with true labels

    TODO:
        Consider doing distributions in space
    '''
    # true labels shape is time, and subsetted to a specific region
    # pred labels shape is forecast_hour, time
    # Both are already averaged down

    if var_name in full_names.keys():
        variable = full_names[var_name]
    else:
        variable = 'Overall'

    true_dataset = [true, true, true, true]
    pred_dataset = [pred[:,0], pred[:,29], pred[:,59], pred[:,89]]

    # Make the plot
    fig, ax = plt.subplots(figsize = [12, 8])

    # Set the title
    ax.set_title('%s Anomalies [%s]'%(variable, units[var_name]), fontsize = 18)

    # Make the plot
    true_pcs = ax.violinplot(true_dataset, showmeans = True, showextrema = True,
                             side = 'low')#, label = 'True Anomalies')
    pred_pcs = ax.violinplot(pred_dataset, showmeans = True, showextrema = True, 
                             side = 'high')#, label = 'Predicted Anomalies')

    for pc in true_pcs['bodies']:
        pc.set_facecolor(anomaly_plotting_information[var_name]['color_true'])
        pc.set_edgecolor(anomaly_plotting_information[var_name]['color_true'])
        pc.set_alpha(0.6)
        
    for pc in pred_pcs['bodies']:
        pc.set_facecolor(anomaly_plotting_information[var_name]['color_pred'])
        pc.set_edgecolor(anomaly_plotting_information[var_name]['color_pred'])
        pc.set_alpha(0.6)

    true_pcs['cmeans'].set_color(anomaly_plotting_information[var_name]['color_true'])
    true_pcs['cmeans'].set_linestyle('--')
    true_pcs['cmeans'].set_linewidth(3)

    pred_pcs['cmeans'].set_color(anomaly_plotting_information[var_name]['color_pred'])
    pred_pcs['cmeans'].set_linestyle('--')
    pred_pcs['cmeans'].set_linewidth(3)

    true_pcs['cmaxes'].set_color(anomaly_plotting_information[var_name]['color_true'])
    true_pcs['cmins'].set_color(anomaly_plotting_information[var_name]['color_true'])
    true_pcs['cbars'].set_color(anomaly_plotting_information[var_name]['color_true'])
    true_pcs['cmaxes'].set_linewidth(3)
    true_pcs['cmins'].set_linewidth(3)

    pred_pcs['cmaxes'].set_color(anomaly_plotting_information[var_name]['color_pred'])
    pred_pcs['cmins'].set_color(anomaly_plotting_information[var_name]['color_pred'])
    pred_pcs['cbars'].set_color(anomaly_plotting_information[var_name]['color_pred'])
    pred_pcs['cmaxes'].set_linewidth(3)
    pred_pcs['cmins'].set_linewidth(3)

    # Make the legend
    labels = []
    true_color_patch = mpatches.Patch(color = true_pcs['bodies'][0].get_facecolor().flatten())
    pred_color_patch = mpatches.Patch(color = pred_pcs['bodies'][0].get_facecolor().flatten())
    labels.append((true_color_patch, 'True Labels'))
    labels.append((pred_color_patch, 'CrossFormer'))
    ax.legend(*zip(*labels), loc = 'upper left', fontsize = 18)

    # Add labels
    ax.set_ylabel('%s Anomalies [%s]'%(variable, units[var_name]), fontsize = 18)

    ax.set_xlabel('Lead Time [Days]', fontsize = 18)

    # Ticks
    fh = [1, 30, 60, 90]
    ax.set_xticks(np.arange(1, len(fh)+1))

    ax.set_xticklabels(fh)

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
    extend = ['max', 'both']

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
        # axes[n].set_title(labels[n], size = 18)
        axes[n].set_title(title_information, size = 18)
        
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
        cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmaps[n], extend = extend[n], orientation = 'vertical')
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
    extend = ['max', 'both']
    
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
        # axes[n].set_title(labels[n], size = 16)
        axes[n].set_title(title_information, size = 16)
        
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
        cbax = fig.add_axes([horizontal_coords[n], 
                             subset_information[subset]['colorbar_coord_metric'][0],
                             subset_information[subset]['colorbar_coord_metric'][1],
                             subset_information[subset]['colorbar_coord_metric'][2]])#0.330, 0.34, 0.015])
        cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmaps[n], extend = extend[n], orientation = 'horizontal')
        cbar.ax.set_xlabel(color_labels[n], fontsize = 16)

        # Set the colorbar ticks
        for i in cbar.ax.xaxis.get_ticklabels():
            i.set_size(16)
        
        
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

# Histogram scatter plot
def make_histogram_scatter_plot(ytrue, ypred, variable, forecast_day, path = './Figures', savename = 'tmp.png'):

    valid_indices = np.invert(np.isnan(ytrue.flatten()) | np.isnan(ypred.flatten()))

    full_variable_name = full_names[variable]

    cmin = 0; cmax = 30; cint = (cmax - cmin)/100
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'viridis', lut = nlevs)

    # max_value = np.nanmax(ytrue)
    # min_value = np.nanmin(ytrue)
    max_value = color_information[variable]['hist_values'][1]
    min_value = color_information[variable]['hist_values'][0]
    max_counts = color_information[variable]['max_counts']

    # Certain variables should have minimums at 0
    if ('swvl' in variable) | ('fdii' in variable) | ('ws' in variable) | ('fg10' in variable) | ('tp' in variable):
        min_value = 0

    # Make the least squares line
    slope, intercept = least_squares(ypred.flatten()[valid_indices], ytrue.flatten()[valid_indices])
    xhat = np.linspace(min_value, max_value, 100)
    yhat = slope * xhat + intercept

    # Make the plot
    fig, ax = plt.subplot_mosaic([['hist_pred', '.'], ['hist2d', 'hist_true']], figsize = [12, 12],
                                  width_ratios = (4, 1), height_ratios = (1, 4))
    # fig, ax = plt.subplots(figsize = [12, 12])
    plt.subplots_adjust(hspace = 0.05, wspace = 0.05)

    # Set the title
    fig.suptitle('2D Histogram for %s\nfor a %02d day forecast'%(full_variable_name, forecast_day), y = 0.93, fontsize = 18)

    ax['hist_pred'].hist(ypred.flatten()[valid_indices], bins = 100, color = cmap(0.25))
    ax['hist_pred'].tick_params(axis = 'x', labelbottom = False)
    ax['hist_pred'].set_ylim([0, max_counts])
    ax['hist_pred'].set_xlim([min_value, max_value])
    for i in ax['hist_pred'].yaxis.get_ticklabels():
        i.set_size(18)

    ax['hist_true'].hist(ytrue.flatten()[valid_indices], bins = 100, color = cmap(0.25), orientation = 'horizontal')
    ax['hist_true'].tick_params(axis = 'y', labelleft= False)
    ax['hist_true'].tick_params(axis = 'x', labelrotation = 270)
    ax['hist_true'].set_xlim([0, max_counts])
    ax['hist_true'].set_ylim([min_value, max_value])
    for i in ax['hist_true'].xaxis.get_ticklabels():
        i.set_size(18)
    
    # Make the plot; # Note the * 100 turns it from a fraction of the whole to a percentage
    h2d = ax['hist2d'].hist2d(ypred.flatten()[valid_indices], ytrue.flatten()[valid_indices], bins = 100, 
                              cmin = cmin+0.001, cmax = None, 
                              norm = mcolors.Normalize(vmin=cmin+1e-5, vmax = cmax, clip = True), cmap = cmap)
    print(np.nanmin(h2d[0]), np.nanmax([h2d[0]]), np.nanmean(h2d[0]))

    ax['hist2d'].plot([min_value, max_value], [min_value, max_value], color = 'r', linewidth = 2.5, linestyle = '-')
    ax['hist2d'].plot(xhat, yhat, color = 'k', linestyle = '-', linewidth = 2.5, label = 'y = %4.2fx + %4.2f'%(slope, intercept))
    ax['hist2d'].legend(loc = 'upper left', fontsize = 18)

    # Add labels
    ax['hist2d'].set_xlabel('Predicted Labels (%s)'%units[variable], fontsize = 18)
    ax['hist2d'].set_ylabel('True labels (%s)'%units[variable], fontsize = 18)
    ax['hist2d'].set_xlim([min_value, max_value])
    ax['hist2d'].set_ylim([min_value, max_value])

    # Tick information
    for i in ax['hist2d'].xaxis.get_ticklabels() + ax['hist2d'].yaxis.get_ticklabels():
        i.set_size(18)

    # cbax = fig.add_axes([0.91, 0.11, 0.018, 0.77]) # for vertical orientation
    cbax = fig.add_axes([0.13, 0.03, 0.60, 0.018]) # for horizontal orientation
    cbar = mcolorbar.Colorbar(cbax, mappable = h2d[3], orientation = 'horizontal')
    cbar.ax.set_xlabel('Counts', fontsize = 16)
    for i in cbar.ax.xaxis.get_ticklabels():
        i.set_size(18)
    #cb = fig.colorbar(h2d, ax = ax, label = 'Counts')

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


def make_histogram(metric, variable, metric_name, path = './Figures/', savename = 'timeseries.png'):
    '''
    Display a line histogram
    '''

    full_variable_name = full_names[variable]

    valid_indices = np.invert(np.isnan(metric.flatten()))
    plot_metric = metric.flatten()[valid_indices]

    # Make the plot
    fig, ax = plt.subplots(figsize = [12, 8])

    # Set the title
    # ax.set_title('Distribution of %s for %s'%(metric_name.upper(), full_variable_name), fontsize = 18)
    ax.set_title('%s'%(full_variable_name), fontsize = 18)
    
    # Make the plot; # Note the * 100 turns it from a fraction of the whole to a percentage
    bins = ax.hist(plot_metric, bins = 80, weights = 100 * np.ones((plot_metric.size))/plot_metric.size, 
                   histtype = 'step', log = True, color = 'black', linewidth = 1.0)


    # Plot climatology and persistence (note these include forecasts at day = 0, so start from 1)
    # if climatology is not None:
    #     ax.plot(y, climatology, color = 'g', linewidth = 2.5, label = 'Climatology')

    # if persistence is not None:
    #     ax.plot(y, persistence, color = 'b', linewidth = 2.5, label = 'Persistence')

    # if (climatology is not None) | (persistence is not None):
    #     ax.legend(fontsize = 18)
        
    # Add labels
    ax.set_ylabel('Fraction of pixels at error (%)', fontsize = 18)
    ax.set_xlabel('%s (%s)'%(metric_name.upper(), units[variable]), fontsize = 18)

    # Tick information
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(18)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_metric_error_plots(ytrue, files, time, lat, lon,
                            sname, plot_sname, subset, variable, apply_mask, data_path, figure_path):
    '''
    '''

    # Need to get sname and args.variable and lat and lon and times
    time = np.array([t.to_pydatetime() for t in time])

    y_pred = []
    time_pred = []

    # Load the latitude weights
    with Dataset('%s/lat_and_lons.nc'%data_path, 'r') as nc:
        tmp = nc.variables['latitude'][:]
        tmp = np.cos(tmp*np.pi/180)

    # exampe weights to the same shape as the variable data
    latitude_weights = np.ones((ytrue.shape[1], ytrue.shape[2]))
    for j in range(latitude_weights.shape[-1]):
        latitude_weights[:,j] = tmp

    # Load in the predictions
    for file in new_sort(files):
        with Dataset(file, 'r') as nc:
            time_pred_tmp = nc.variables['time'][:]
            time_pred.append(datetime(1900,1,1) + timedelta(hours = time_pred_tmp.item()))
            
            if ('200 mb' in variable) | ('200_mb' in variable):
                y_pred.append(nc.variables[sname][0,0,:,:])
            elif ('500 mb' in variable) | ('500_mb' in variable):
                y_pred.append(nc.variables[sname][0,1,:,:])
            else:
                y_pred.append(nc.variables[sname][0,:,:])

    y_pred = np.array(y_pred)
    
    time_ind = np.where( (time >= time_pred[0]) & (time <= time_pred[-1]) )[0]
    ytrue = ytrue[time_ind,:,:]

    # Calculate the error metrics
    acc = calculate_acc_in_space(y_pred, ytrue, latitude_weights = latitude_weights)
    rmse = calculate_rmse_in_space(y_pred, ytrue, latitude_weights = latitude_weights)


    # make the plots
    if subset is not None:
        make_error_subset_map(acc, rmse, lat, lon, subset,
                              var_name = plot_sname,
                              path = figure_path, savename = '%s_%s_error_map.png'%(subset, variable))
        
        if apply_mask:
            # with Dataset('%s/aridity_mask.nc'%data_path, 'r') as nc:
            with Dataset('%s/aridity_mask_reduced.nc'%data_path, 'r') as nc:
                # ai_mask = nc.variables['aim'][:,:-1,:] # Note the aridity mask also masks out the oceans
                ai_mask = nc.variables['aim'][:]
            
            # with Dataset('%s/land.nc'%data_path, 'r') as nc:
            with Dataset('%s/land_reduced.nc'%data_path, 'r') as nc:
                # lsm_mask = nc.variables['lsm'][:,:-1,:] # Includes other sea points not in the aridity mask (e.g., Great Lakes)
                lsm_mask = nc.variables['lsm'][:]

            # Apply the masks
            acc = np.where(ai_mask == 1, acc, np.nan)
            acc = np.where(lsm_mask == 1, acc, np.nan)

            rmse = np.where(ai_mask == 1, rmse, np.nan)
            rmse = np.where(lsm_mask == 1, rmse, np.nan)
            
        make_histogram(rmse, plot_sname, 'rmse', path = figure_path, savename = '%s_%s_rmse_histogram.png'%(subset, variable))

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

def calculate_percentiles(data, timeseries, times, Nmax, months, days):
    '''
    Calculate a sequence (max, 98th, 95th, 90th, 80th, 70th, 50th, 30th, 20th, 10th, 5th, 2nd, min) of percentiles for a time series
    '''

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
    for t in range(times.size):
        # Get the indices for all times with the current date (up to t to avoid end of window errors)
        ind = np.where( (times[t].month == months[n-1:]) & (times[t].day == days[n-1:]) )[0]

        if t > (times.size - Nmax): # (ind_t[0]-1)):
            # Add to the data (N-day accumulation)
            data[t] = data[t] + data[t-1]

            # Conduction a running summation of N days for statistics to compare data_plot to
            tmp = running_sum(timeseries, n)

            n = n + 1
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


    # Turn the statistics into arrays and reverse them so they go forward in time 
    # Note the reversing step is not necessary, it just makes them consistent with data_plot and time_plot
    percentiles_100 = np.array(percentiles_100)
    percentiles_98 = np.array(percentiles_98)
    percentiles_95 = np.array(percentiles_95)
    percentiles_90 = np.array(percentiles_90)
    percentiles_80 = np.array(percentiles_80)
    percentiles_70 = np.array(percentiles_70)
    percentiles_50 = np.array(percentiles_50)
    percentiles_30 = np.array(percentiles_30)
    percentiles_20 = np.array(percentiles_20)
    percentiles_10 = np.array(percentiles_10)
    percentiles_5 = np.array(percentiles_5)
    percentiles_2 = np.array(percentiles_2)
    percentiles_0 = np.array(percentiles_0)

    return data, percentiles_100, percentiles_98, percentiles_95, percentiles_90, percentiles_80, percentiles_70, percentiles_50, percentiles_30, percentiles_20, percentiles_10, percentiles_5, percentiles_2, percentiles_0

def make_drought_fingerprint_plot(timeseries, timeseries_prediction, times, 
                                  start, variable = 'precipitation', region = 'east africa'):
    '''
    Make the drought fingerprint plot

    Fix ticks and colors
    '''
    # Get the variable short name
    sname = get_var_shortname(variable)
    ndays = [90, 180]
    
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

    # Convert the time series to float64 to avoid rounding and precision errors that cause unstable summations
    timeseries = timeseries.astype(np.float64)


    # Calculate accumulations for the predictions
    prediction_plot = timeseries_prediction
    for t in range(prediction_plot.size):
        if t == 0: #(prediction_plot.size - 1):
            continue
        else:
            prediction_plot[t] = prediction_plot[t] + prediction_plot[t-1]

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

        # Calculate accumulation and percentiles
        (data, percentiles_100, percentiles_98, percentiles_95, percentiles_90, 
         percentiles_80, percentiles_70, percentiles_50, percentiles_30, percentiles_20,
         percentiles_10, percentiles_5, percentiles_2, percentiles_0) = calculate_percentiles(data_plot.copy(), timeseries, times_plot, ndays[m], months, days)

        # Plot the statistics
        ax[m].plot(times_plot, percentiles_100, color = 'white', label = 'maximum')
        ax[m].fill_between(times_plot, percentiles_98, np.nanmax(percentiles_100), color = colors['W4'], alpha = 0.9, label = 'W4')
        ax[m].fill_between(times_plot, percentiles_95, percentiles_98, color = colors['W3'], alpha = 0.9, label = 'W3')
        ax[m].fill_between(times_plot, percentiles_90, percentiles_95, color = colors['W2'], alpha = 0.9, label = 'W2')
        ax[m].fill_between(times_plot, percentiles_80, percentiles_90, color = colors['W1'], alpha = 0.9, label = 'W1')
        ax[m].fill_between(times_plot, percentiles_70, percentiles_80, color = colors['W0'], alpha = 0.9, label = 'W0')
        ax[m].plot(times_plot, percentiles_50, color = 'black', label = 'median')
        ax[m].fill_between(times_plot, percentiles_20, percentiles_30, color = colors['D0'], alpha = 0.9, label = 'D0')
        ax[m].fill_between(times_plot, percentiles_10, percentiles_20, color = colors['D1'], alpha = 0.9, label = 'D1')
        ax[m].fill_between(times_plot, percentiles_5, percentiles_10, color = colors['D2'], alpha = 0.9, label = 'D2')
        ax[m].fill_between(times_plot, percentiles_2, percentiles_5, color = colors['D3'], alpha = 0.9, label = 'D3')
        ax[m].fill_between(times_plot, 0, percentiles_2, color = colors['D4'], alpha = 0.9, label = 'D4')
        ax[m].plot(times_plot, percentiles_0, color = 'white', label = 'minimum')

        # Plot the actual data
        ax[m].plot(times_plot, data, color = 'black', linewidth = 3.0, label = 'True Labels')

        # Plot the predictions of the variable for the S2S plot only (since predictions are only 90 days out)
        if 'S2S' in titles[m]:
            times_prediction = times_plot[-prediction_plot.size:]
            ax[m].plot(times_prediction, prediction_plot, color = 'black', linewidth = 3.0, linestyle = '--', label = 'CrossFormer')

        # Make a secondary axis to show days of accumulated variable
        sec_xaxis = ax[m].secondary_xaxis(location = -0.07, 
                                          functions = (partial(datetimes_to_accum_days, initial_day = start), 
                                                       partial(accum_days_to_datetimes, initial_day = start)))

        sec_xaxis.xaxis.set_ticks(secondary_ticks[m])#, labels = secondary_ticks[m].astype(str).tolist())

        #sec_xaxis.grid(True, axis = 'x', color = 'black', linewidth = 2, linestyle = ':')
        # A work around secondary_xaxis.grid() plotting nothing; manually make a line for each tick on the main axis
        for line in sec_xaxis.get_xticks():
            if (line == 0) | (line == ndays[m]):# | (line == ndays[1]):
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
        ax[m].set_xlim([start - timedelta(days = ndays[m]), start])
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


def make_score_cards(base_score, prediction_score, months, variable = 'swvl1', score = 'rmse', subset = None,
                     cmin = -15, cmax = 15, cint = 1, path = './', savename = 'tmp_scorecard.png'):
    '''
    '''

    if subset is None:
        label = 'Global'
    elif subset == 'sh':
        # label = 'Southern Hemisphere'
        label = 'SH'
    elif subset == 'nh':
        # label = 'Northern Hemisphere'
        label = 'NH'
    else:
        label = subset.title()
    
    Nrows = 4 # Number of rows in the table (Base prediction, all months, summer and sprint months, fall and winter months)
    Nprediction_steps = 10 # Number of lead times to show
    y_labels = ['Climatology',
                label, 
                #'CrossFormer\nAll Months', 
                'CrossFormer\nMAMJJA Months', 
                'CrossFormer\nSONDJF Months']
    inds = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89]

    T, Nforecast_days = base_score.shape

    ### Create the matrix to hold displayed values.
    display_values = np.ones((Nrows, Nprediction_steps))

    display_values[0,:] = np.nanmean(base_score, axis = 0)[inds]#[::int(Nforecast_days/Nprediction_steps)] # Base predictions have 0% change/improvement since that is the base level
    # Percent change relative to all years
    display_values[1,:] = np.nanmean(prediction_score, axis = 0)[inds]#[::int(Nforecast_days/Nprediction_steps)]

    # Percent change for all spring and summer months
    ind_summer = np.where( (months >= 3) & (months <= 8) )[0]
    display_values[2,:] = np.nanmean(prediction_score[ind_summer,:], axis = 0)[inds]#[::int(Nforecast_days/Nprediction_steps)]

    # Percent change for all fall and winter months
    ind_winter = np.where( (months >= 9) | (months <= 2) )[0]
    display_values[3,:] = np.nanmean(prediction_score[ind_winter,:], axis = 0)[inds]#[::int(Nforecast_days/Nprediction_steps)]


    ### Create the matrix to make the table of.
    table_plot = np.ones((Nrows, Nprediction_steps))

    table_plot[0,:] = 0 # Base predictions have 0% change/improvement since that is the base level
    # Percent change relative to all years
    table_plot[1,:] = ((display_values[1,:] - display_values[0,:]) / display_values[0,:]) * 100
    # table_plot[1,:] = np.nanmean(percent_change, axis = 0)[inds]#[::int(Nforecast_days/Nprediction_steps)]

    # Percent change for all spring and summer months
    table_plot[2,:] = ((display_values[2,:] - display_values[0,:]) / display_values[0,:]) * 100
    # table_plot[2,:] = np.nanmean(percent_change[ind_summer,:], axis = 0)[inds]#[::int(Nforecast_days/Nprediction_steps)]

    # Percent change for all fall and winter months
    table_plot[3,:] = ((display_values[3,:] - display_values[0,:]) / display_values[0,:]) * 100
    # table_plot[3,:] = np.nanmean(percent_change[ind_winter,:], axis = 0)[inds]#[::int(Nforecast_days/Nprediction_steps)]

    # For squared error, improvement is lower scores (reverse change for lower values is improvement)
    table_plot = table_plot if score == 'acc' else -1 * table_plot

    # Color information
    # cmin = -15; cmax = 15; cint = 1
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)-2

    if nlevs % 2 == 0:
        ncolors = int(np.floor(nlevs/2))
        grays = [(0.95, 0.95, 0.95)]
    else:
        ncolors = int(np.floor(nlevs/2))
        grays = [(0.95, 0.95, 0.95), (0.95, 0.95, 0.95)]

    reds = sns.color_palette('Reds_r', ncolors)
    blues = sns.color_palette('Blues', ncolors)
 
    cmap = mcolors.ListedColormap(reds + grays + blues, name = 'rb_scorecard')
    # norm = mcolors.BoundaryNorm(clevs, cmap.N, extend = 'both')

    # cmap = plt.get_cmap('Spectral', lut = nlevs) #Another color being 'coolwarm'

    # Make the table
    fig, axes = plt.subplots(figsize = [20, 4], nrows = Nrows, ncols = 1)
    plt.subplots_adjust(hspace = 0.09)

    for row in range(Nrows):
        ax = axes[row]

        # The matrix must be 2D for imshow
        im_plot = np.zeros((1, Nprediction_steps))
        im_plot[0,:] = table_plot[row,:]

        if row == 0:
            im = ax.imshow(im_plot, vmin = -4, vmax = 1, cmap = 'gray', 
                           interpolation = 'nearest')
        else:
            im = ax.imshow(im_plot, vmin = cmin, vmax = cmax, cmap = cmap, 
                           interpolation = 'nearest')


        # Add values to the table
        thresh = cmin * 8.0/10 #(np.nanmax(table_plot) + np.nanmin(table_plot))/2
        for j in range(Nprediction_steps):
            # Color the text
            # color = im.cmap(nlevs) if table_plot[i, j] < thresh else im.cmap(0)
            color = 'white' if np.abs(table_plot[row, j]) > np.abs(thresh) else 'black' # if np.abs(table_plot[row, j]) <= 5 else 'white'

            # Add the text
            if score == 'acc':
                text = format(display_values[row, j], '4.2f')
            elif np.nanmean(display_values[row, j]) < 1:
                text = format(display_values[row, j], '0.2f')
            else:
                text = format(display_values[row, j], '0.3g')
            ax.text(j, 0, text, ha = 'center', va = 'center', color = color, fontsize = 16)

            # Add a white border between cells
            rect = plt.Rectangle((j-0.5, 0-0.5), 1, 1, fill = False, color = 'white', linewidth = 1.5)
            ax.add_patch(rect)
        # for i, j in product(range(Nrows), range(Nprediction_steps)):
        #     # Color the text
        #     # color = im.cmap(nlevs) if table_plot[i, j] < thresh else im.cmap(0)
        #     color = 'black' if np.abs(table_plot[i, j]) <= 5 else 'white'

        #     # Add the text
        #     # text = format(base_score[i, j], '0.3g') if i == 0 else format(prediction_score[i, j], '0.3g')
        #     text = format(display_values[i, j], '0.3g')
        #     ax.text(j, i, text, ha = 'center', va = 'center', color = color, fontsize = 14)

        # Set the ticks
        if row == (Nrows - 1):
            ax.set_xticks(np.arange(Nprediction_steps))

            # Tick labels

            xticks = np.arange(0, Nforecast_days+1, Nprediction_steps)
            xticks[0] = xticks[0] + 1
            ax.set_xticklabels(xticks, fontsize = 16)

            ax.set_xlabel('Lead Time [days]', fontsize = 16)

        else:
            ax.set_xticks([])

            ax.set_xticklabels('')
        
        ax.set_yticks(np.arange(1))

        ax.set_yticklabels([y_labels[row]], fontsize = 16)

        # Set the spines/grid lines betweehn values
        for spine in ax.spines.values():
            spine.set_edgecolor('#808080')
            spine.set_linewidth(2.0)

        
    if score == 'acc':
        fig.suptitle('%s %s'%(full_names[variable], score.upper()), fontsize = 16)
    else:
        fig.suptitle('%s %s (%s)'%(full_names[variable], score.upper(), units[variable]), fontsize = 16)

    # Set the colorbar
    cbax = fig.add_axes([0.320, -0.10, 0.385, 0.045])

    # Add the colorbar
    cbar = fig.colorbar(im, cax = cbax, extend = 'both', orientation = 'horizontal')

    interval = 2 if score == 'acc' else 3

    # Set the ticks
    cbar.set_ticks(np.round(np.arange(cmin, cmax + cint, interval)))
    cbar.set_label(r'Worse $\leftarrow$ %% difference in %s vs Climatology $\rightarrow$ Better'%score.upper(), fontsize = 16)
    for i in cbar.ax.xaxis.get_ticklabels():
        i.set_size(14)

    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


def plot_ffts(true_labels, predicted_labels, hour = 24):
    '''
    Plot make and plot the Fourier transforms for a truth and prediction dataset.
    '''
    # Make the fast Fourier transforms (FFTs)
    true_ffts = fft.fft(true_labels.flatten())
    predicted_ffts = fft.fft(predicted_labels.flatten())

    true_spectrum = np.real(true_ffts)**2 + np.imag(true_ffts)**2
    predicted_spectrum = np.real(predicted_ffts)**2 + np.imag(predicted_ffts)**2


    # Make the plot
    fig, ax = plt.subplots(figsize = [10, 10])
    ax.plot(true_spectrum[1:101], 'r', linewidth = 1.5, label = 'True Labels')
    ax.plot(predicted_spectrum[1:101], 'b', linewidth = 1.5, label = 'Predicted Labels')
    ax.legend(loc = 'upper right')

    ax.set_ylabel('Power Spectra', fontsize = 14)
    ax.set_xlabel('Wavenumber', fontsize = 14)

    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(14)

    plt.savefig('spatial_power_spectra_%02d_day_forecast.png'%int(hour/24))


def make_comparison_subset_map_2(y, y_pred, climatology, lat, lon, time, subset,
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
    empty_formatter = cticker.LatitudeFormatter(cardinal_labels = {'north': '', 'south': ''})
    
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
    fig, axes = plt.subplots(figsize = [18, 18], nrows = 1, ncols = 4, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(wspace = subset_information[subset]['wspace'], hspace = subset_information[subset]['hspace'])
    for n, data in enumerate([y, climatology, y_pred]):
        # Set the title
        axes[n].set_title('%s\nValid for %s'%(labels[n], time.isoformat()), size = 14)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        axes[n].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

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

            axes[n].set_xticklabels(LonLabel, fontsize = 14)

            axes[n].xaxis.set_major_formatter(LonFormatter)

            if n == 0:
                axes[n].set_yticks(LatLabel, crs = data_proj)

                axes[n].set_yticklabels(LatLabel, fontsize = 14)
                
                axes[n].yaxis.set_major_formatter(LatFormatter)
            else:
                axes[n].yaxis.set_major_formatter(empty_formatter)

        # Plot the data
        cs = axes[n].pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                                cmap = cmap, transform = data_proj, zorder = 1)

        # Set the map extent
        axes[n].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])


    # Set the colorbar size and location
    if (var_name == 'ndvi') | (var_name == 'fpar') | (var_name == 'cvh') | (var_name == 'cvl'):
        extend = None
    elif ('tp' in var_name) | ('swvl' in var_name) | ('fdii' in var_name) | ('q_tot' in var_name) | (var_name == 'ws') | (var_name == 'fg10') | (var_name == 'ssr'):
        extend = 'max'
    else:
        extend = 'both'

    # cbax = fig.add_axes([0.925, 0.365, 0.020, 0.26]) # Settings for a single row
    cbax = fig.add_axes(subset_information[subset]['colorbar_coord'][0])#[0.925, 0.535, 0.020, 0.21])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = extend, orientation = 'horizontal')
    cbar.ax.set_xlabel('%s (%s)'%(variables[var_name], units[var_name]), fontsize = 14)

    # Set the colorbar ticks
    for i in cbar.ax.xaxis.get_ticklabels():
        i.set_size(14)
        


    # Make the true - pred plot
    cmin_new = anomaly_plotting_information[var_name]['difference_climits'][0]; cmax_new = anomaly_plotting_information[var_name]['difference_climits'][1]
    cint = (cmax_new - cmin_new)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin_new, cmax_new + cint, cint)
    nlevs = len(clevs)
    cmap_new = plt.get_cmap(name = 'coolwarm_r', lut = nlevs)

    axes[3].set_title('True - Predicted Labels', size = 14)

    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    axes[3].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Ocean covers and "masks" data outside the U.S.
    axes[3].coastlines(edgecolor = 'black', zorder = 3)
    axes[3].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

    # Adjust the ticks
    if fig_proj == ccrs.EckertIII():
        gl = axes[3].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                draw_labels = True,
                                x_inline = False, y_inline = False,
                                lw = 0.8, linestyle = '--', color = 'grey')
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        gl.rotate_labels = False
    else:
        axes[3].set_xticks(LonLabel, crs = data_proj)
        # axes[3].set_yticks(LatLabel, crs = data_proj)

        # axes[3].set_yticklabels(LatLabel, fontsize = 14)
        axes[3].set_xticklabels(LonLabel, fontsize = 14)

        axes[3].xaxis.set_major_formatter(LonFormatter)
        axes[3].yaxis.set_major_formatter(empty_formatter)

    # Plot the data
    cs = axes[3].pcolormesh(lon, lat, y - y_pred, vmin = cmin_new, vmax = cmax_new,
                              cmap = cmap_new, transform = data_proj, zorder = 1)

    # Set the map extent
    axes[3].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    cbax = fig.add_axes(subset_information[subset]['colorbar_coord'][1])#[0.925, 0.235, 0.020, 0.21])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap_new, extend = 'both', orientation = 'horizontal')
    cbar.ax.set_xlabel('Difference (%s)'%(units[var_name]), fontsize = 14)

    # Set the colorbar ticks
    for i in cbar.ax.xaxis.get_ticklabels():
        i.set_size(14)
    
        
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()


def make_anomaly_subset_map_2(y, y_pred, lat, lon, time, subset,
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
    empty_formatter = cticker.LatitudeFormatter(cardinal_labels = {'north': '', 'south': ''})
    
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
    fig, axes = plt.subplots(figsize = [18, 18], nrows = 1, ncols = 3, 
                             subplot_kw = {'projection': fig_proj})
    plt.subplots_adjust(wspace = subset_information[subset]['wspace'], hspace = subset_information[subset]['hspace_anomaly'])
    for n, data in enumerate([y, y_pred]):
        # Set the title
        axes[n].set_title('%s\nValid for %s'%(labels[n], time.isoformat()), size = 16)
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        axes[n].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

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

            axes[n].set_xticklabels(LonLabel, fontsize = 14)

            axes[n].xaxis.set_major_formatter(LonFormatter)

            if n == 0:
                axes[n].set_yticks(LatLabel, crs = data_proj)

                axes[n].set_yticklabels(LatLabel, fontsize = 14)
                
                axes[n].yaxis.set_major_formatter(LatFormatter)
            else:
                axes[n].yaxis.set_major_formatter(empty_formatter)

        # Plot the data
        cs = axes[n].pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                                cmap = cmap, transform = data_proj, zorder = 1)

        # Set the map extent
        axes[n].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])


    # Set the colorbar size and location
    # cbax = fig.add_axes([0.925, 0.365, 0.020, 0.26]) # Settings for a single row
    cbax = fig.add_axes(subset_information[subset]['colorbar_coord_anomaly'][0])#[0.925, 0.565, 0.020, 0.26])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = 'both', orientation = 'horizontal')
    cbar.ax.set_xlabel('%s Anomaly (%s)'%(variables[var_name], units[var_name]), fontsize = 16)

    # Set the colorbar ticks
    for i in cbar.ax.xaxis.get_ticklabels():
        i.set_size(16)
        
    # Make the true - pred plot
    cmin_new = anomaly_plotting_information[var_name]['difference_climits'][0]; cmax_new = anomaly_plotting_information[var_name]['difference_climits'][1]
    cint = (cmax_new - cmin_new)/20

    #cmin = 0; cmax = np.round(np.nanmax(np.stack([y, y_pred]))*1.10, 1); cint = (cmax - cmin)/20
    clevs = np.arange(cmin_new, cmax_new + cint, cint)
    nlevs = len(clevs)
    cmap_new = plt.get_cmap(name = 'coolwarm_r', lut = nlevs)
    axes[2].set_title('True - Predicted Anomalies', size = 16)

    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    axes[2].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Ocean covers and "masks" data outside the U.S.
    axes[2].coastlines(edgecolor = 'black', zorder = 3)
    axes[2].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

    # Adjust the ticks
    if fig_proj == ccrs.EckertIII():
        gl = axes[2].gridlines(xlocs = LonLabel, ylocs = LatLabel,
                                draw_labels = True,
                                x_inline = False, y_inline = False,
                                lw = 0.8, linestyle = '--', color = 'grey')
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        gl.rotate_labels = False
    else:
        axes[2].set_xticks(LonLabel, crs = data_proj)
        # axes[2].set_yticks(LatLabel, crs = data_proj)

        # axes[2].set_yticklabels(LatLabel, fontsize = 14)
        axes[2].set_xticklabels(LonLabel, fontsize = 14)

        axes[2].xaxis.set_major_formatter(LonFormatter)
        axes[2].yaxis.set_major_formatter(empty_formatter)

    # Plot the data
    cs = axes[2].pcolormesh(lon, lat, y - y_pred, vmin = cmin_new, vmax = cmax_new,
                              cmap = cmap_new, transform = data_proj, zorder = 1)

    # Set the map extent
    axes[2].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    cbax = fig.add_axes(subset_information[subset]['colorbar_coord_anomaly'][1])#[0.925, 0.165, 0.020, 0.26])
    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap_new, extend = 'both', orientation = 'horizontal')
    cbar.ax.set_xlabel('Difference (%s)'%(units[var_name]), fontsize = 16)
    cbar.ax.set_xticks(np.arange(cmin_new, cmax_new+1, 5))

    # Set the colorbar ticks
    for i in cbar.ax.xaxis.get_ticklabels():
        i.set_size(16)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()

if __name__ == '__main__':
    # Currently just a test for the drought plot
    import zarr
    import numpy as np
    import imageio
    from datetime import datetime, timedelta
    from netCDF4 import Dataset
    from glob import glob
    # from plotting import make_drought_fingerprint_plot

    make_score_card = True
    make_spectra_plot = False
    create_fingerprint_plot = False

    if make_score_card:
        import pandas as pd
        from utils import get_metric_information, new_sort

        metrics = {}
        climatology = {}
        persistence = {}
        n = 0

        start_date = datetime(2001, 1, 1); Ndays = 365
        dates = np.array([start_date + timedelta(days = day) for day in range(Ndays)])
        months = np.array([date.month for date in dates])

        # Determine the path
        data_path = '%s/forecasts/metrics'%'/ourdisk/hpc/ai2es/sedris/results/rotation_00/'

        # Collect all files to load
        files = glob('%s/2023*T00Z_africa.csv'%(data_path), recursive = True)
        clim_files = glob('%s/climatology/rotation_1/africa_clim_2001-*.nc'%('/ourdisk/hpc/ai2es/sedris/results'), recursive = True)
        persist_files = glob('%s/persistence/rotation_01/africa_persist_2001-*.nc'%('/ourdisk/hpc/ai2es/sedris/results'), recursive = True)

        # Select out only those climatology and persistence files that correspond to the rotation being loaded
        # ind = [np.where(date == all_dates)[0] for date in rotation_dates]
        # clim_files = new_sort(clim_files)[ind]
        # persist_files = new_sort(persist_files)[ind]

        # Load the metric data + climatology and persistence data
        for m, file in enumerate(new_sort(files)[:150]):
            metrics[n] = pd.read_csv(file, sep = ',',
                                    index_col = 0, 
                                    header = 0,
                                    nrows = 90)
    
            
            # Load climatology metrics
            with Dataset(clim_files[m], 'r') as nc:
                climatology[m] = {}
                for key in nc.variables.keys():
                    climatology[m][key] = nc.variables[key][:]

            # Load persistence metrics
            with Dataset(persist_files[n], 'r') as nc:
                persistence[m] = {}
                for key in nc.variables.keys():
                    persistence[n][key] = nc.variables[key][:]

            n = n + 1

        # Reorganize data to time x forecast day
        metrics_names = []
        metric_list = {}
        clim_list = {}
        persist_list = {}
        percent_change = {}
        for metric in metrics[0].keys():
            if metric in ['time', 'latitude', 'longitude', 'forecast_step', 'datetime']: # These include time and forecast_step, which is what the metrics are plotted against
                continue
            
            # Collect individual metric information
            metric_name, var_name, level = get_metric_information(metric)

            # Convert dictionary information of a variable and metric into arrays of size N_forecasts x forecast_length
            metric_list[metric] = []
            clim_list[metric] = []
            persist_list[metric] = []
            for key in metrics.keys():
                # Get the metric
                metric_list[metric].append(metrics[key][metric])
                clim_list[metric].append(climatology[key][metric])
                persist_list[metric].append(persistence[key][metric])

            metrics_names.append(metric)
            metric_list[metric] = np.array(metric_list[metric])
            clim_list[metric] = np.array(clim_list[metric])
            persist_list[metric] = np.array(persist_list[metric])
        
        make_score_cards(clim_list['%s_%s'%('rmse', 'swvl1')], metric_list['%s_%s'%('rmse', 'swvl1')], 
                         months[:150], variable = 'swvl1', score = 'rmse', path = './', savename = 'tmp_scorecard.png')

    if make_spectra_plot:
        year = 2018

        hours = np.arange(24, 2160+1, 24)
        start = datetime(2018, 6, 1)
        start_str = start.isoformat()

        root = zarr.open_group('../credit_datasets/surface.%d.zarr'%year)
        # root = zarr.open_group('../credit_datasets/diagnostic.%d.zarr'%year)
        true = root['tp'][:]

        for hour in hours:
            print('Forecast %d day'%int(hour/24))
            with Dataset('../backup_credit/forecasts/%sZ/pred_%sZ_%03d.nc'%(start_str[:-6], start_str[:-6], hour), 'r') as nc:
            # with Dataset('/scratch/rarrell/credit_model_single_step/forecasts/%sZ/pred_%sZ_%03d.nc'%(start_str[:-6], start_str[:-6], hour), 'r') as nc:
                pred = nc.variables['tp'][:]
                print(pred.shape)
            
            plot_ffts(true[int(hour/24),:,:], pred, hour)

        images = []
        figures = glob('spatial_power_spectra_*_day_forecast.png', recursive = True)
        figures = np.sort(figures)
        for figure in figures:
            images.append(imageio.imread(figure))

        # Make the gif
        sname = 'spatial_precipitation_spectra_gif.mp4'
        imageio.mimsave('%s'%(sname), images, format = 'FFMPEG')

    if create_fingerprint_plot:
        variable = []
        # Southwestern US
        # lower_lat = 30
        # upper_lat = 40
        # lower_lon = 240
        # upper_lon = 255
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
        lower_lat = -5
        upper_lat = 20
        lower_lon = 35
        upper_lon = 52

        years = np.arange(2000, 2024)

        for year in years:
            print(year)
            root = zarr.open_group('../credit_datasets/surface.%d.zarr'%year)
            # root = zarr.open_group('../credit_datasets/diagnostic.%d.zarr'%year)
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
            with Dataset('../backup_credit/forecasts/%sZ/pred_%sZ_%03d.nc'%(start_str[:-6], start_str[:-6], hour), 'r') as nc:
            # with Dataset('/scratch/rarrell/credit_model_single_step/forecasts/%sZ/pred_%sZ_%03d.nc'%(start_str[:-6], start_str[:-6], hour), 'r') as nc:
                tmp = np.nanmean(nc.variables['tp'][:])
                # tmp = np.nanmean(nc.variables['swvl1'][:])
                prediction.append(tmp)
        prediction = np.array(prediction)

        make_drought_fingerprint_plot(variable, prediction, times, start, variable = 'precipitation', region = 'Eastern Africa')
