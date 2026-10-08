"""Provides a set of functions designed to facilitate
the calculation of the Standardized Evaporative Stress
Ratio (SESR) and Flash Drought Intensity Index (FDII)
for flash drought (FD) monitoring and prediction
"""

import gc
import numpy as np
import pandas as pd

from scipy import stats
from scipy import interpolate
from scipy import signal
from scipy import special
from netCDF4 import Dataset
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob
from datetime import datetime, timedelta
from typing import Tuple, Union

# from inputs_outputs import load_nc

def calculate_climatology(
        e, 
        pet, 
        dates_all, 
        days_per_year: int = 366
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculates the climatological mean and standard deviation of ESR from daily ERA5 data.
    Climatological data is calculated for all grid points and for all timestamps in the year.

    Inputs:
    :param e: Evaporation dataset (np.ndarray, with shape time x lat x lon)
    :param pet: Potential evaporation dataset (np.ndarray with shape time x lat x lon)
    :param dates_all: Datetimes labels for each time step in e and pet (np.ndarray with shape time)
    :param days_per_year: Total number of days in one year of data (use 366 if using daily data to include leap day)

    Outputs:
    :param means: Mean of ESR for each grid and date in year (np.ndarray of shape time_for_one_year x lat x lon)
    :param stds: Standard deviations for each grid and date in year (np.ndarray of shape time_for_one_year x lat x lon)
    :param one_year: Datetime labels for each time step in means and stds (np.ndarray of shape time_for_one_year)
    '''
    

    T, I, J = e.shape
    T = days_per_year # Numbers of days in a year

    # All years in climatology calculations
    all_years = np.unique([date.year for date in dates_all])
    years = np.array([date.year for date in dates_all])

    days = np.array([day.day for day in dates_all])
    months = np.array([day.month for day in dates_all])

    # Get datetimes for one year (includes leap day)
    dates_year = np.array([datetime(2012,1,1) + timedelta(days = day) for day in range(days_per_year)])

    # Initialize climatology means and standard deviations + counts
    means = np.zeros((T, I, J), dtype = np.float32)
    stds = np.zeros((T, I, J), dtype = np.float32)

    N = np.zeros((T))

    # Construct ESR
    esr = e/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = np.nan
    esr[esr > 3] = np.nan 
    # print(np.nansum(np.isnan(esr)))

    print('Initialized variables, calculation means')

    # Conduct climatology calculations
    for t, date in enumerate(dates_year):
        # Get all days in the current date in the loop
        ind = np.where( (date.day == days) & (date.month == months) )[0]

        # Sum over all all ESR in a given day
        tmp_sum = np.nansum(esr[ind,:,:], axis = 0)
        means[t,:,:] = np.nansum([means[t,:,:], tmp_sum], axis = 0) # np.nansum to account for any NaNs
        N[t] = N[t] + len(ind)

    # At the end, loop again to get to divide the sums by N to get the means
    for t, date in enumerate(dates_year):
        means[t,:,:] = means[t,:,:]/N[t]

    means = means.astype(np.float32)

    print('Means calculated, calculating standard deviations')
    
    # Loop over each day in the year for standard deviation (requires means)
    for t, date in enumerate(dates_year):
        # Get all days in the current date in the loop
        ind = np.where( (date.day == days) & (date.month == months) )[0]
        
        # Sum over the squared errors in a given date
        error = np.nansum((esr[ind,:,:] - means[t,:,:])**2, axis = 0)
        stds[t,:,:] = np.nansum([stds[t,:,:], error], axis = 0)

    # One final loop to finish standard deviation calculations
    for t, date in enumerate(dates_year):
        stds[t,:,:] = np.sqrt(stds[t,:,:]/(N[t] - 1))

    stds = stds.astype(np.float32)
    print(np.min(means), np.max(means))
    print(np.min(stds), np.max(stds))

    print('Standard deviations calculated')

    # Create datetime labels for each timestep in the means and standard deviations
    ind = np.where(years == 2012)[0]
    one_year = dates_all[ind]

    return means, stds, one_year

def calculate_sesr(
        et, 
        pet, 
        dates, 
        means, 
        stds, 
        one_year
        ) -> np.ndarray:
    '''
    Calculate the standardized evaporative stress ratio (SESR) from ET and PET.
    
    Full details on SESR can be found in Christian et al. 2019 (for SESR): https://doi.org/10.1175/JHM-D-18-0198.1.
    
    Inputs:
    :param et: Evapotranspiration (ET) dataset (np.ndarray of shape time x lat x lon)
    :param pet: Potential evapotranspiration (PET) dataset (np.ndarray of shape time x lat x lon)
    :param dates: Datetime labels for each timestep in et and pet (np.ndarray of shape time)
    :param means: Mean of ESR for each grid point and date in the year (np.ndarray of shape time_for_one_year x lat x lon)
    :param stds: Standard deviation of ESR for each grid point and date in the year (np.ndarray of shape time_for_one_year x lat x lon)
    :param one_year: Datetime labels for each time step in means and stds (np.ndarray of shape time_for_one_year)
        
    Outputs:
    :param sesr: Calculate SESR (np.ndarray with shape time x lat x lon)
    '''

    # Get shape information
    T, I, J = et.shape
    # dates = np.array([datetime(year, 1, 1) + timedelta(days = t) for t in range(T)])


    # Obtain the evaporative stress ratio (ESR); the ratio of ET to PET
    esr = et/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = np.nan
    esr[esr > 3] = np.nan
    # print(np.nansum(np.isnan(esr)))

    # Collect date information
    months = np.array([date.month for date in one_year])
    days = np.array([date.day for date in one_year])

    # Initialize SESR
    sesr = np.ones((T, I, J)) * np.nan

    for t, date in enumerate(dates):
        # Find the date index for the one year range
        ind = np.where( (date.month == months) & (date.day == days) )[0]
        
        # Standardize the ESR to get SESR
        sesr[t,:,:] = (esr[t,:,:] - means[ind[0],:,:])/stds[ind[0],:,:]
            

    # Remove any unrealistic points
    sesr = np.where(sesr < -5, -5, sesr)
    sesr = np.where(sesr > 5, 5, sesr)

    sesr = sesr.astype(np.float32)
    
    print(np.nanmin(sesr), np.nanmax(sesr), np.nanmean(sesr))
    # print(np.sum(sesr <= -4.5), np.nansum(sesr >= 4.5))
    return sesr

def calculate_fdii(
        smp, 
        dates, 
        apply_runmean = True, 
        mask = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculate the flash drought intensity index (FDII) from a soil moisture percentiles.
    FDII is on the same time scale as the input data.
    
    Full details on FDII can be found in Otkin et al. 2021: https://doi.org/10.3390/atmos12060741

    Note FDII can be calculated with the standardized soil moisture, or percentiles.
    Percentiles are used here for consistancy with Otkin et al. 2021
    
    Inputs:
    :param smp: Soil moisture percentile dataset (np.ndarray with shape time x lat x lon)
    :param year: Datetime labels for each timestep in smp (np.ndarray of shape time)
    :param apply_runmean: Apply a centered running mean (length 5) to the percentiles before FDII calculations (recommended for daily data)
    :param use_mask: Indicates whether to use a land-sea mask to improve computation speed
    :param mask: Land-sea mask with values 1 for land and 0 for sea (np.ndarray with shape lat x lon)

    Outputs:
    :param fdii: FDII drought index (np.ndarray with shape time x lat x lon)
    :param fd_int: The strength of the rapid intensification of the flash drought (np.ndarray with shape time x lat x lon)
    :param dro_sev: Severity of the drought component of the flash drought (np.ndarray with shape time x lat x lon)
    '''
       
    print('Initializing some variables')
    # Define some base constants
    PER_BASE = 15 # Minimum percentile drop for FD is 15 percentiles in 4 pentads
    T_BASE   = 4*5
    DRO_BASE = 20 # Percentiles must be below the 20th percentile to be in drought
    
    T, I, J = smp.shape

    # Make the years, months, and/or days variables
    years = np.array([date.year for date in dates])
    months = np.array([date.month for date in dates])
    days = np.array([date.day for date in dates])

    # Apply a 5 day running mean requested by the user
    if apply_runmean:
        print('Applying 5 day running mean')
        runmean = 5

        # Determine the appropriate start and end index for a centered running mean 
        start_ind = int(np.round((runmean - 1)/2))
        end_ind = int(T + runmean - 1 - start_ind)

        # Apply running mean for each grid point
        for i in tqdm(range(I), desc = 'Applying running mean'):
            for j in range(J):
                smp[:,i,j] = np.convolve(smp[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]
    
    
    print(np.nanmin(smp), np.nanmax(smp))
    print(np.nanmean(smp))
    
    print('Calculating rapid intensification of flash drought')
    # Determine the rapid intensification based on percentile changes 
    # based on equation 1 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
    fd_int = np.zeros((T, I, J))

    # Determine the intensification index
    # Note many time related values are multiplied by 5 to correspond to daily data instead of pentad
    for i in tqdm(range(I), desc = 'Calculating FD_INT'):
        for j in range(J):
            # Ignore sea points
            if mask[i,j] == 0: # ERA5 only
                continue
        
            for t in range(T-10): # Note the last two pentads are excluded as there is not enough time for significant SM drop
            
                obs = np.zeros((9*5)) # Note, the method detailed in Otkin et al. 2021 involves looking ahead 2 to 10 pentads (9 entries total)
                for nday in np.arange(2*5, 10*5+5, 1):
                    nday = int(nday)
                    if (t+nday) >= T: # If t + npend is in the future (beyond the dataset), break the loop and use 0s for obs instead
                        break         # This should only effect results in one November and December
                    else:
                        obs[nday-10] = (smp[t+nday,i,j] - smp[t,i,j])/nday # Note npend is the number of pentads the system is currently looking ahead to.

                # If the maximum change in percentiles is less than the base change requirement (15 percentiles in 4 pentads), set FD_INT to 0.
                #  Otherwise, determine FD_INT according to eq. 1 in Otkin et al. 2021
                if np.nanmax(obs) < (PER_BASE/T_BASE):
                    fd_int[t,i,j] = 0
                else:
                    fd_int[t,i,j] = ((PER_BASE/T_BASE)**(-1)) * np.nanmax(obs)
                
    print(np.min(fd_int), np.max(fd_int), np.mean(fd_int))
    
    
    print('Calculating drought severity')
    # Next determine the drought severity component using equation 2 in Otkin et al. 2021 
    # (and detailed in section 2.2 of the same paper)
    dro_sev = np.zeros((T, I, J)) # Initialize the first entry to 0, since there is no rapid intensification before it

    for i in tqdm(range(I), desc = 'Calculating DRO_SEV'):
        for j in range(J):
            
            # Ignore sea values
            if mask[i,j] == 0: # ERA5 only
                continue
            
            for t in range(1, T-5):
                if (fd_int[t,i,j] > 0):
                    
                    dro_sum = 0
                    for nday in np.arange(0, 18*5+5, 1): # In Otkin et al. 2021, the DRO_SEV can look up to 18 pentads (90 days) in the future for its calculation
                        
                        if (t+nday) >= T:      # For simplicity, set DRO_SEV to 0 when near the end of the dataset 
                            dro_sev[t,i,j] = 0 # (this should only impact results near the end of the dataset)
                            break
                        else:
                            dro_sum = dro_sum + (DRO_BASE - smp[t+nday,i,j])
                            
                            if smp[t+nday,i,j] > DRO_BASE: # Terminate the summation and calculate DRO_SEV if SM is no longer below the base percentile for drought
                                if nday < 4*5:
                                    # DRO_SEV is set to 0 if drought was not consistent for at least 4 pentads after rapid intensificaiton (i.e., little to no impact)
                                    dro_sev[t,i,j] = 0
                                    break
                                else:
                                    dro_sev[t,i,j] = dro_sum/nday # Terminate the loop and determine the drought severity if the drought condition is broken
                                    break
                                
                            elif (nday >= 18*5): # Calculate the drought severity of the loop goes out 90 days, but the drought does not end
                                dro_sev[t,i,j] = dro_sum/nday
                                break
                            else:
                                pass
                
                # In continuing consistency with Otkin et al. 2021, if the pentad does not immediately follow rapid intensification, drought is set 0
                else:
                    dro_sev[t,i,j] = 0
    
    print(np.min(dro_sev), np.max(dro_sev), np.mean(dro_sev))
    
    print('Calculating FDII')
    
    # Finally, FDII is the product of the rapid intensification and drought severity components
    fdii = fd_int * dro_sev
    
    print(np.min(fdii), np.max(fdii), np.mean(fdii))

    # Remove values less than 0
    fd_int[fd_int <= 0] = 0
    dro_sev[dro_sev <= 0] = 0
    fdii[fdii <= 0] = 0

    print('Done')
    
    return fdii, fd_int, dro_sev


def calculate_sm_percentiles(
        sm, 
        sm_all, 
        dates, 
        dates_all, 
        mask = None, 
        ) -> np.ndarray:
    '''
    Calculate the soil moisture percentiles using a larger popularion of soil moisture data

    Note this method is NOT space efficient. It requires in the full soil moisture dataset which 
    can be large to produce timely computations.

    Inputs:
    :param sm: Soil moisture dataset (np.ndarray with shape time x lat  x lon)
    :param sm_all: Full soil moisture dataset 
                   (list, with each list entry being one year of sm data, an np.ndarray with shape time_for_one_year x lat x lon)
    :param dates: Datetime labels for each time step in sm (np.ndarray with shape time)
    :param dates_all: Datetime labels for each time step in sm_all (np.ndarray with shape time_for_all_dates)
    :param mask: Land-sea mask with values 1 for land and 0 for sea (np.ndarray with shape lat x lon)
    
    Outputs:
    :param smp: Percentiles for each value in sm (np.ndarray with shape time x lat x lon)
    '''
        
    T, I, J= sm.shape # Obtain the dataset size to intialize the percentile dataset
    
    # All years in the time series
    all_years = np.unique([date.year for date in dates_all])

    # Calculate all years in the full time series
    years = np.array([date.year for date in dates_all])
    months = np.array([date.month for date in dates_all])
    days = np.array([date.day for date in dates_all])

        
    # Initialize percentile dataset
    smp = np.zeros((T, I, J))
 
    # sm = np.concatenate(sm, axis = 0)

    # n = 0
    for i in tqdm(range(I)):
        for j in range(J):
            # Skip sea values
            if mask[i,j] == 0:
                continue

            #print('%d/%d'%(n, I*J))
            sm_time_series = []

            # Determine the complete time series for a given grid point
            for y, _ in enumerate(all_years):
                sm_time_series.append(sm_all[y][:,i,j])
            
            sm_time_series = np.concatenate(sm_time_series)
            
            # Calculate the SM percentiles for all points in the time axis for a given grid point
            for t, date in enumerate(dates):
                # Obtain all indices for the current day of the year
                ind = np.where((date.day == days) & (date.month == months))[0] 

                # Calculate the SM percentile based on the current day of the year
                smp[t,i,j] = stats.percentileofscore(sm_time_series[ind], sm[t,i,j])

            # n = n+1

    # print(np.nansum(smp <10), np.nansum(smp > 90))
    smp = smp.astype(np.float32)

    return smp

def _apply_running_mean_3d(data, runmean=5, sample=None, verbose=True):
    '''
    Apply a centered running mean along the time axis for each grid point.
    Modifies data (and sample if provided) in place.
    '''
    T, I, J = data.shape

    if verbose:
        print('Applying 5 day running mean')

    # Centered convolution: trim edges so output length matches input time dimension
    start_ind = int(np.round((runmean - 1)/2))
    end_ind = int(T + runmean - 1 - start_ind)

    end_ind_sample = None
    if sample is not None:
        end_ind_sample = int(sample.shape[0] + runmean - 1 - start_ind)

    for i in tqdm(range(I), desc='Applying running mean', disable=np.invert(verbose)):
        for j in range(J):
            data[:, i, j] = np.convolve(data[:, i, j], np.ones((runmean))/runmean)[start_ind:end_ind]
            if sample is not None:
                sample[:, i, j] = np.convolve(sample[:, i, j], np.ones((runmean))/runmean)[start_ind:end_ind_sample]

    return data, sample


def _compute_fd_thresholds(
        sesr_filt_sample,
        delta_sesr_sample,
        one_year,
        months,
        days,
        climo_index,
        include_intensity,
        sesr_threholds=None,
        dsesr_percentile=25,
        d2_percentile=20,
        d3_percentile=15,
        d4_percentile=10,
        sesr_percentile=20,
        verbose=True,
        ):
    '''
    Compute or load climatological flash-drought thresholds for Christian et al. FD method.
    '''
    _, I, J = sesr_filt_sample.shape

    if sesr_threholds is None:
        dc_crit = np.ones((366, I, J)) * np.nan
        ri_crit = np.ones((366, I, J)) * np.nan
        if include_intensity:
            d2_crit = np.ones((366, I, J)) * np.nan
            d3_crit = np.ones((366, I, J)) * np.nan
            d4_crit = np.ones((366, I, J)) * np.nan

        # Day-of-year climatological percentiles for drought component and rapid intensification
        for t, date in tqdm(enumerate(one_year)):
            ind = np.where((date.month == months[climo_index]) & (date.day == days[climo_index]))[0]

            dc_crit[t, ...] = np.nanpercentile(sesr_filt_sample[ind, ...], sesr_percentile, axis=0)
            ri_crit[t, ...] = np.nanpercentile(delta_sesr_sample[ind, ...], dsesr_percentile, axis=0)
            if include_intensity:
                d2_crit[t, ...] = np.nanpercentile(delta_sesr_sample[ind, ...], d2_percentile, axis=0)
                d3_crit[t, ...] = np.nanpercentile(delta_sesr_sample[ind, ...], d3_percentile, axis=0)
                d4_crit[t, ...] = np.nanpercentile(delta_sesr_sample[ind, ...], d4_percentile, axis=0)

    else:
        dc_crit = sesr_threholds[0].astype(np.float32)
        ri_crit = sesr_threholds[1].astype(np.float32)
        if include_intensity:
            d2_crit = sesr_threholds[2].astype(np.float32)
            d3_crit = sesr_threholds[3].astype(np.float32)
            d4_crit = sesr_threholds[4].astype(np.float32)

    if include_intensity:
        return dc_crit, ri_crit, d2_crit, d3_crit, d4_crit
    return dc_crit, ri_crit


# FD definitions
def christian_fd(
        sesr, 
        mask, 
        dates, 
        include_intensity = False,
        start_year = 1990, 
        end_year = 2020, 
        apply_runmean = False,
        sesr_sample = None,
        save_thresholds = False,
        years = None, 
        months = None, 
        days = None,
        sesr_threholds = None,
        verbose = True,
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    '''
    Calculate flash drought using an updated Christian et al. standardized evaporative 
    stress ratio (SESR)-based method.

    Christian et al. 2019: https://doi.org/10.1175/JHM-D-18-0198.1.
    Updates for LSWI in Christian et al. 2022: https://doi.org/10.1016%2Fj.rsase.2022.100770.

    Inputs:
    :param sesr: Input SESR values (np.ndarray with shape time x lat x lon)
    :param mask: Land-sea mask (1 = land, 0 = sea; shape lat x lon). Pass None to disable masking.
    :param dates: Datetime labels for each timestep in sesr (np.ndarray with shape time)
    :param include_intensity: If True, label FD intensity categories 1–4; if False, binary 0/1 (default False)
    :param start_year: First year of the climatological period for threshold computation (default 1990)
    :param end_year: Last year of the climatological period for threshold computation (default 2020)
    :param apply_runmean: Apply a centered 5-day running mean to SESR before FD calculations (default False)
    :param sesr_sample: Extended SESR record for SG filtering and climatological thresholds
                        (np.ndarray with shape time x lat x lon). Required when sesr time length
                        is shorter than the Savitzky-Golay window (105 days).
    :param save_thresholds: If True, return computed threshold arrays instead of FD labels (default False)
    :param years: Array of integers for dates.year. If None, derived from dates.
    :param months: Array of integers for dates.month. If None, derived from dates.
    :param days: Array of integers for dates.day. If None, derived from dates.
    :param sesr_threholds: Precomputed threshold tuple (dc_crit, ri_crit [, d2_crit, d3_crit, d4_crit]).
                           If None, thresholds are computed from sesr_sample.
    :param verbose: If True, print progress messages (default True)
    
    Outputs:
    :param fd: Flash drought labels (np.ndarray with shape time x lat x lon).
               Values are 0 (no FD), 1 (D1), and optionally 2–4 when include_intensity=True.
               If save_thresholds=True, returns threshold array(s) instead.
    '''
    
    # Make the years, months, and/or days variables?
    if years is None:
        years = np.array([date.year for date in dates])
        
    if months is None:
        months = np.array([date.month for date in dates])
        
    if days is None:
        days = np.array([date.day for date in dates])
        
    T, I, J = sesr.shape

    # Is a mask provided?
    mask_provided = mask is not None
        
    if apply_runmean:
        sesr, sesr_sample = _apply_running_mean_3d(sesr, runmean=5, sample=sesr_sample, verbose=verbose)

    # Initialize some variables
    sesr_inter = np.ones((T, I, J)) * np.nan
    sesr_filt  = np.ones((T, I, J)) * np.nan
    
    climo_index = np.where( (years >= start_year) & (years <= end_year) )[0]
    
    # sesr = sesr.reshape(T, I*J, order = 'F')
    # sesr_inter = sesr_inter.reshape(T, I*J, order = 'F')
    # sesr_filt  = sesr_filt.reshape(T, I*J, order = 'F')
    
    # if mask_provided:
    #     mask = mask.reshape(I*J, order = 'F')
    
    x = np.arange(-6.5, 6.5, (13/T))#[:-1] # a variable covering the range of all SESR values with 1 entry for each time step
    if x.size > T:
        x = x[:-1]

    if sesr_sample is not None:
        x_sample= np.arange(-6.5, 6.5, (13/sesr_sample.shape[0]))
        sesr_inter_sample = np.ones((sesr_sample.shape[0], I, J)) * np.nan
        sesr_filt_sample  = np.ones((sesr_sample.shape[0], I, J)) * np.nan

    # Parameters for the Savitzky-Golay filter
    # 21 pentads x 5 days/pentad = 105-day window for daily data
    WinLength = 21*5
    PolyOrder = 4

    if T <= WinLength:
        # Treat the special case when the examined SESR data is shorter than the window length.

        # NOTE: This is a special case that REQUIRES a sample dataset to be provided to apply the SG filter and determine percentiles

        # Make a copy of the sample data
        sesr_copy = sesr_sample.copy() # Need to reduce the window length for prediction timescales

        # Replace the desired spot with the SDL predictions
        dates_tmp = np.array([datetime(year, month, day) for (year, month, day) in zip(years, months, days)])
        replace_ind = np.where((dates_tmp >= dates[0]) & (dates_tmp <= dates[-1]))[0]

        sesr_copy[replace_ind,...] = sesr
        sesr = sesr_copy

        # Re-initialize some variables with the new time length
        sesr_inter = np.ones((sesr.shape[0], I, J)) * np.nan
        sesr_filt  = np.ones((sesr.shape[0], I, J)) * np.nan
        x = np.arange(-6.5, 6.5, (13/sesr.shape[0]))

    # Perform a basic linear interpolation for NaN values and apply a SG filter
    if verbose:
        print('Applying interpolation and Savitzky-Golay filter to SESR')
    for i in tqdm(range(I), desc = 'Applying SG Filter', disable = np.invert(verbose)):
        for j in range(J):
            if mask_provided:
                if mask[i,j] == 0:
                    continue
                else:
                    pass
            
            # Perform a linear interpolation to remove NaNs
            ind = np.isfinite(sesr[:,i,j])

            if np.nansum(ind) == 0:
                continue
            else:
                pass
            
            ind = np.where(ind == True)[0]
            interp_func = interpolate.interp1d(x[ind], sesr[ind,i,j], kind = 'linear', fill_value = 'extrapolate')
            
            sesr_inter[:,i,j] = interp_func(x)
            
            # Apply the Savitzky-Golay filter to the interpolated SESR data
            sesr_filt[:,i,j] = signal.savgol_filter(sesr_inter[:,i,j], WinLength, PolyOrder)

            # Repeat with the sample size if necessary
            if sesr_sample is not None:
                ind_sample = np.isfinite(sesr_sample[:,i,j])

                ind_sample = np.where(ind_sample == True)[0]
                interp_func = interpolate.interp1d(x_sample[ind_sample], sesr_sample[ind_sample,i,j], kind = 'linear', fill_value = 'extrapolate')
                
                sesr_inter_sample[:,i,j] = interp_func(x_sample)
                
                # Apply the Savitzky-Golay filter to the interpolated SESR data
                sesr_filt_sample[:,i,j] = signal.savgol_filter(sesr_inter_sample[:,i,j], WinLength, PolyOrder)
        
    # Reorder SESR back to 3D data
    #sesr_filt = sesr_filt.reshape(T, I, J, order = 'F')
    if T <= WinLength:
        # For small SESR, re-obtain the predicted SESR values
        sesr_filt = sesr_filt[replace_ind,...]
        del sesr_copy, sesr_inter_sample, sesr_inter; gc.collect()

    if sesr_sample is None:
        sesr_filt_sample = sesr_filt

    # Determine the change in SESR
    if verbose:
        print('Calculating the change in SESR')

    delta_sesr  = np.ones((T, I, J)) * np.nan
    delta_sesr_sample  = np.ones((sesr_filt_sample.shape[0], I, J)) * np.nan
    
    delta_sesr[1:,:,:] = sesr_filt[1:,:,:] - sesr_filt[:-1,:,:]
    delta_sesr_sample[1:,:,:] = sesr_filt_sample[1:,:,:] - sesr_filt_sample[:-1,:,:]
    
    # Begin the flash drought calculations
    if verbose:
        print('Identifying flash drought')
    fd = np.ones((T, I, J)) * np.nan

    fd = fd.astype(np.float32)
    delta_sesr = delta_sesr.astype(np.float32)
    delta_sesr_sample = delta_sesr_sample.astype(np.float32)
    sesr_filt = sesr_filt.astype(np.float32)
    sesr_filt_sample = sesr_filt_sample.astype(np.float32)

    #fd = fd.reshape(T, I*J, order = 'F')
    #sesr_filt = sesr_filt.reshape(T, I*J, order = 'F')
    #delta_sesr = delta_sesr.reshape(T, I*J, order = 'F')

    # Christian et al. climatological threshold percentiles (drought component and rapid intensification)
    dsesr_percentile = 25
    d2_percentile = 20
    d3_percentile = 15
    d4_percentile = 10
    sesr_percentile  = 20
    
    min_change = timedelta(days = 30)
    start_date = dates[-1]

    one_year = np.array([datetime(2012, 1, 1) + timedelta(days = day) for day in range(366)])
    months_year = np.array([date.month for date in one_year])
    days_year = np.array([date.day for date in one_year])

    thresholds = _compute_fd_thresholds(
        sesr_filt_sample,
        delta_sesr_sample,
        one_year,
        months,
        days,
        climo_index,
        include_intensity,
        sesr_threholds=sesr_threholds,
        dsesr_percentile=dsesr_percentile,
        d2_percentile=d2_percentile,
        d3_percentile=d3_percentile,
        d4_percentile=d4_percentile,
        sesr_percentile=sesr_percentile,
        verbose=verbose,
    )
    if include_intensity:
        dc_crit, ri_crit, d2_crit, d3_crit, d4_crit = thresholds
    else:
        dc_crit, ri_crit = thresholds

    if save_thresholds:
        return (dc_crit, ri_crit, d2_crit, d3_crit, d4_crit) if include_intensity else (dc_crit, ri_crit)

    if False:
        fd = fd.reshape(T, I*J)
        sesr_filt = sesr_filt.reshape(T, I*J)
        delta_sesr = delta_sesr.reshape(T, I*J)

        dc_crit = dc_crit.reshape(dc_crit.shape[0], I*J)
        ri_crit = ri_crit.reshape(ri_crit.shape[0], I*J)
        if include_intensity:
            d2_crit = d2_crit.reshape(d2_crit.shape[0], I*J)
            d3_crit = d3_crit.reshape(d3_crit.shape[0], I*J)
            d4_crit = d4_crit.reshape(d4_crit.shape[0], I*J)

        start_date = np.array([dates[-1] for _ in range(I*J)])
        if include_intensity:
            start_date_d2 = np.array([dates[-1] for _ in range(I*J)])
            start_date_d3 = np.array([dates[-1] for _ in range(I*J)])
            start_date_d4 = np.array([dates[-1] for _ in range(I*J)])

        def update_dates(start_dates, crit):
            IJ = start_dates.size

            for ij in range(IJ):
                if (delta_sesr[t,ij] <= crit[ij]) & (start_dates[ij] == dates[-1]):
                    start_dates[ij] = dates[t]
                elif (delta_sesr[t,ij] <= crit[ij]) & (start_dates[ij] != dates[-1]):
                    pass
                else:
                    start_dates[ij] = dates[-1]

        for t in tqdm(range(T), desc = 'Identifying FD', disable = np.invert(verbose)):
            ind = np.where( (dates[t].month == months_year) & (dates[t].day == days_year) )[0]
            if include_intensity:
                fd[t,:] = np.where(( (dates[t] - start_date) >= min_change) & (sesr_filt[t,:] <= dc_crit[ind,:]), 1, 0)
                fd[t,:] = np.where(( (dates[t] - start_date_d2) >= min_change) & (sesr_filt[t,:] <= dc_crit[ind,:]), 2, fd[t,:])
                fd[t,:] = np.where(( (dates[t] - start_date_d3) >= min_change) & (sesr_filt[t,:] <= dc_crit[ind,:]), 3, fd[t,:])
                fd[t,:] = np.where(( (dates[t] - start_date_d4) >= min_change) & (sesr_filt[t,:] <= dc_crit[ind,:]), 4, fd[t,:])
            else:
                fd[t,:] = np.where(( (dates[t] - start_date) >= min_change) & (sesr_filt[t,:] <= dc_crit[ind,:]), 1, 0)

            if include_intensity:
                update_dates(start_date_d4, d4_crit[ind[0],:])
                update_dates(start_date_d3, d3_crit[ind[0],:])
                update_dates(start_date_d2, d2_crit[ind[0],:])
                update_dates(start_date, ri_crit[ind[0],:])
            else:
                update_dates(start_date, ri_crit[ind[0],:])

        fd = fd.reshape(T, I, J)
    
    for i in tqdm(range(I), desc = 'Identifying FD', disable = np.invert(verbose)):
        for j in range(J):
            if mask_provided:
                if mask[i,j] == 0:
                    continue
            
            start_date = dates[-1]
            if include_intensity:
                start_date_d2 = start_date_d3 = start_date_d4 = dates[-1]
            for t in range(T):
                # ind = np.where( (dates[t].month == months[climo_index]) & (dates[t].day == days[climo_index]) )[0]
                ind = np.where( (dates[t].month == months_year) & (dates[t].day == days_year) )[0]
                
                # # Determine the percentiles of dSESR and SESR
                # ri_crit = np.nanpercentile(delta_sesr_sample[ind,i,j], dsesr_percentile)
                # dc_crit = np.nanpercentile(sesr_filt_sample[ind,i,j], sesr_percentile)

                # if include_intensity:
                #     d4_crit = np.nanpercentile(delta_sesr_sample[ind,i,j], d4_percentile)
                #     d3_crit = np.nanpercentile(delta_sesr_sample[ind,i,j], d3_percentile)
                #     d2_crit = np.nanpercentile(delta_sesr_sample[ind,i,j], d2_percentile)

                
                # If start_date != dates[-1], the rapid intensification criteria is satisfied
                # If the rapid intensification and drought component criteria are satisfied (and FD period is 30+ days)
                # then FD occurs
                if include_intensity:
                    if ( (dates[t] - start_date_d4) >= min_change) & (sesr_filt[t,i,j] <= dc_crit[ind,i,j]):
                        fd[t,i,j] = 4
                    elif ( (dates[t] - start_date_d3) >= min_change) & (sesr_filt[t,i,j] <= dc_crit[ind,i,j]):
                        fd[t,i,j] = 3
                    elif ( (dates[t] - start_date_d2) >= min_change) & (sesr_filt[t,i,j] <= dc_crit[ind,i,j]):
                        fd[t,i,j] = 2
                    elif ( (dates[t] - start_date) >= min_change) & (sesr_filt[t,i,j] <= dc_crit[ind,i,j]):
                        fd[t,i,j] = 1
                    else:
                        fd[t,i,j] = 0
                else:
                    if ( (dates[t] - start_date) >= min_change) & (sesr_filt[t,i,j] <= dc_crit[ind,i,j]):
                        fd[t,i,j] = 1
                    else:
                        fd[t,i,j] = 0
                
                # If the change in SESR is below the criteria, change the start date of the flash drought
                if include_intensity:
                    if (delta_sesr[t,i,j] <= d4_crit[ind,i,j]) & (start_date_d4 == dates[-1]):
                        start_date_d4 = dates[t]
                    elif (delta_sesr[t,i,j] <= d4_crit[ind,i,j]) & (start_date_d4 != dates[-1]):
                        pass
                    else:
                        start_date_d4 = dates[-1]

                    if (delta_sesr[t,i,j] <= d3_crit[ind,i,j]) & (start_date_d3 == dates[-1]):
                        start_date_d3 = dates[t]
                    elif (delta_sesr[t,i,j] <= d3_crit[ind,i,j]) & (start_date_d3 != dates[-1]):
                        pass
                    else:
                        start_date_d3 = dates[-1]

                    if (delta_sesr[t,i,j] <= d2_crit[ind,i,j]) & (start_date_d2 == dates[-1]):
                        start_date_d2 = dates[t]
                    elif (delta_sesr[t,i,j] <= d2_crit[ind,i,j]) & (start_date_d2 != dates[-1]):
                        pass
                    else:
                        start_date_d2 = dates[-1]

                    if (delta_sesr[t,i,j] <= ri_crit[ind,i,j]) & (start_date == dates[-1]):
                        start_date = dates[t]
                    elif (delta_sesr[t,i,j] <= ri_crit[ind,i,j]) & (start_date != dates[-1]):
                        pass
                    else:
                        start_date = dates[-1]
                else:
                    if (delta_sesr[t,i,j] <= ri_crit[ind,i,j]) & (start_date == dates[-1]):
                        start_date = dates[t]
                    elif (delta_sesr[t,i,j] <= ri_crit[ind,i,j]) & (start_date != dates[-1]):
                        pass
                    else:
                        start_date = dates[-1]
                
                # print(delta_sesr[t,i,j], ri_crit[ind,i,j], start_date)
    
    # Apply the mask
    for t in range(T):
        fd[t,:,:] = np.where(mask == 1, fd[t,:,:], np.nan)
            
    # Re-order the flash drought back into a 3D array
    # fd = fd.reshape(T, I, J, order = 'F')
    if verbose:
        print('Done')
    
    return fd


def yuan_fd(
        smp, 
        mask, 
        dates, 
        include_intensity = False,
        apply_runmean = False, 
        smp_sample = None,
        years = None, 
        months = None, 
        days = None,
        verbose = True,
        ) -> np.ndarray:
    '''
    Calculate flash drought using the Yuan et al. soil moisture percentile method.

    Yuan et al. 2019: https://doi.org/10.1038/s41467-019-12692-7.
    Uses soil moisture percentiles (SMP; typically 0–40 cm average) to identify flash drought.

    Inputs:
    :param smp: Input soil moisture percentiles (np.ndarray with shape time x lat x lon)
    :param mask: Land-sea mask (1 = land, 0 = sea; shape lat x lon). Sea points are skipped.
    :param dates: Datetime labels for each timestep in smp (np.ndarray with shape time)
    :param include_intensity: If True, assign intensity categories 1–4 based on overall SMP change;
                              if False, binary FD labels (default False)
    :param apply_runmean: Apply a centered 5-day running mean to SMP before FD calculations (default False)
    :param smp_sample: Reference SMP population for intensity threshold percentiles
                       (np.ndarray with shape time x lat x lon). If None, smp is used.
    :param years: Array of integers for dates.year. If None, derived from dates.
    :param months: Array of integers for dates.month. If None, derived from dates.
    :param days: Array of integers for dates.day. If None, derived from dates.
    :param verbose: If True, print progress messages and grid dimensions (default True)
    
    Outputs:
    :param fd: Flash drought labels (np.ndarray with shape time x lat x lon).
               Values are 0 (no FD), 1 (D1), and optionally 2–4 when include_intensity=True.
    '''
    
    # Make the years, months, and/or days variables?
    if years is None:
        years = np.array([date.year for date in dates])
        
    if months is None:
        months = np.array([date.month for date in dates])
        
    if days is None:
        days = np.array([date.day for date in dates])
        
    T, I, J = smp.shape

    if apply_runmean:
        smp, smp_sample = _apply_running_mean_3d(smp, runmean=5, sample=smp_sample, verbose=verbose)

        
    # Begin drought identification process
    if verbose:
        print('Identifying flash droughts')
        print(T, I, J)
    fd = np.zeros((T, I, J), dtype = np.float32) * np.nan
    

    d1_perc = 25
    d2_perc = 20
    d3_perc = 15
    d4_perc = 10

    if smp_sample is None:
        smp_sample = smp
  
    for i in tqdm(range(I), desc = 'Determining FD', disable = np.invert(verbose)):
        for j in range(J):
        
            if mask[i,j] == 0:
                continue
            
            for t in range(T-12*5): # Exclude the last few months in the dataset for simplicity since FD identification involves looking up to 12 pentads ahead
                # If FD analysis was already conducted (process involves looking ahead), skip the analysis
                if (fd[t,i,j] >= 1) | (fd[t,i,j] == 0):
                    continue

                # First determine if the soil moisture is below the 40 percentile (FD possibly begins)
                if smp[t,i,j] <= 40:
                    rate = []

                    # Start looping up to 12 pentads (60 days) ahead
                    for p in range(1, 12*5):
                        # Determine the rate of percentile change
                        if (t + p >= T - 1):
                            rate.append(smp[t+p,i,j] - smp[-1,i,j])
                        else:
                            rate.append(smp[t+p-1,i,j] - smp[t+p,i,j])

                        # When the percentiles fall below the 20th percentile, drought begins
                        # Also another requirement for FD is the average range rate of SMP change must be >= 5 percentiles/pentad = 1 percentile/day
                        # (i.e., an average decrease of 1 percentile per day)
                        # print(rate)
                        if (smp[t+p,i,j] <= 20) & (np.nanmean(rate) >= 1):
                            # Continue looking forward to when the SM percentiles are above 20 (drought ends/recover begins)
                            # Note the indices collected should equate to the number of days, after SMP < 20 to when SMP > 20 (only true for daily data)
                            drought_recover = np.where(smp[t+p:,i,j] > 20)[0]
                            # print(smp[t+p:t+12*5,i,j])
                            # print(drought_recover)
                            # print(smp[t+p:,i,j].shape, smp[t+p:,i,j])

                            # Unique case near the end of the time series; since no points were found to end drought at the end,
                            # label all points from t+p onwards as FD
                            if (len(drought_recover) < 1) | ((t+p) > (T - 12*5)):
                                fd[t:t+p,i,j] = 0

                                if include_intensity:
                                    overall_change = smp[t+p,i,j] - smp[t,i,j]

                                    # Determine all points where the same change is
                                    ind_start = np.where((months == dates[t].month) & (days == dates[t].day))[0]
                                    ind_end = np.where((months == dates[t+p].month) & (days == dates[t+p].day))[0]

                                    all_changes = []
                                    for i_s, i_e in zip(ind_start, ind_end):
                                        all_changes.append(smp_sample[i_e,i,j] - smp_sample[i_s,i,j])
                                    
                                    # Determine if the intensity threshold is met
                                    d4_thresh = np.nanpercentile(all_changes, d4_perc)
                                    d3_thresh = np.nanpercentile(all_changes, d3_perc)
                                    d2_thresh = np.nanpercentile(all_changes, d2_perc)
                                    if overall_change <= d4_thresh:
                                        fd[t+p:,i,j] = 4
                                    elif overall_change <= d3_thresh:
                                        fd[t+p:,i,j] = 3
                                    elif overall_change <= d2_thresh:
                                        fd[t+p:,i,j] = 2
                                    else:
                                        fd[t+p:,i,j] = 1
                                
                                # No intensity threshold is examined
                                else:
                                    fd[t+p:,i,j] = 1
                                break

                            # The last requirement for FD: The drought must last for 15+ days
                            # FD ends on the first instance when SM percentiles > 20 (so first entry in drought recovery)
                            if drought_recover[0] >= 15:
                                # Intensification period is labeled as non-FD (FD is still developing)
                                fd[t:t+p,i,j] = 0

                                if include_intensity:
                                    overall_change = smp[t+p,i,j] - smp[t,i,j]

                                    # Determine all points where the same change is
                                    ind_start = np.where((months == dates[t].month) & (days == dates[t].day))[0]
                                    ind_end = np.where((months == dates[t+p]) & (days == dates[t+p]))[0]

                                    all_changes = []
                                    for i_s, i_e in zip(ind_start, ind_end):
                                        all_changes.append(smp_sample[i_e,i,j] - smp_sample[i_s,i,j])
                                    
                                    # Determine if the intensity threshold is met
                                    d4_thresh = np.nanpercentile(all_changes, d4_perc)
                                    d3_thresh = np.nanpercentile(all_changes, d3_perc)
                                    d2_thresh = np.nanpercentile(all_changes, d2_perc)
                                    if overall_change <= d4_thresh:
                                        fd[t+p:t+p+drought_recover[0],i,j] = 4
                                    elif overall_change <= d3_thresh:
                                        fd[t+p:t+p+drought_recover[0],i,j] = 3
                                    elif overall_change <= d2_thresh:
                                        fd[t+p:t+p+drought_recover[0],i,j] = 2
                                    else:
                                        fd[t+p:t+p+drought_recover[0],i,j] = 1
                                
                                # No intensity threshold is examined
                                else:
                                    # Label all days when SM percentiles < 20 as FD
                                    fd[t+p:t+p+drought_recover[0],i,j] = 1

                                # Analysis is concluded; break the loop that is looking ahead 
                                # (all time points in this event should be labeled FD)
                                break

                            elif drought_recover[0] < 15:
                                # Event was too short to be impactful and thus classified as FD
                                fd[t:t+p+drought_recover[0],i,j] = 0
                        
                        # If drought condition is reached, but the decline was not rapid enough for FD, 
                        # the event drought is labeled as non-FD
                        elif (smp[t+p,i,j] <= 20) & (np.nanmean(rate) < 1):
                            drought_recover = np.where(smp[t+p:,i,j] > 20)[0]

                            # Unique case near the end of the time series; simply break the loop 
                            # (leftover NaNs will be turned to 0 later)
                            if (len(drought_recover) < 1) | ((t+p) > (T - 12*5)):
                                break

                            fd[t:t+p+drought_recover[0],i,j] = 0

                            # Conclude analysis for current event
                            break

                # SM percentiles above the 40th percentile (no occurrence of FD)
                else:
                    fd[t,i,j] = 0
        
        # print(np.nansum(fd[:,i,:]))
    
    # Sea values, and remaining days in the last 60 days of the dataset are labeled as non-FD
    fd[np.isnan(fd)] = 0

    # Apply the mask
    for t in range(T):
        fd[t,:,:] = np.where(mask == 1, fd[t,:,:], np.nan)

    if verbose:
        print('Done')
    
    return fd
