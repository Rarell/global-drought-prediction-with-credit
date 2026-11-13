'''
Drought and flash drought calculation functions.

Implements SESR, soil moisture percentiles, Christian/Yuan/Otkin flash drought
definitions, FDII, SPI, and SPEI following published methodologies.
'''

# Imports
import os, sys, warnings
import numpy as np
import pandas as pd

import gc
from scipy import stats
from scipy import interpolate
from scipy import signal
from scipy import special
from tqdm import tqdm
from glob import glob
from datetime import datetime, timedelta
from typing import Tuple, Union

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


# Drought definition
def calculate_spi(
    precipitation, 
    dates, 
    save_parameters = False, 
    distribution_parameters = None, 
    verbose = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    '''
    Calculate the standardized precipitation index (SPI) from daily precipitation data.

    Full details on SPI can be found in McKee et al. 1993 and the WMO SPI user guide.

    Inputs:
    :param precipitation: Daily precipitation dataset (np.ndarray of shape time x lat x lon)
    :param dates: Datetime labels for each timestep in precipitation (np.ndarray of shape time)
    :param save_parameters: If True, return fitted gamma distribution parameters instead of SPI values
    :param distribution_parameters: Pre-computed gamma parameters as (alpha, beta) arrays of shape time_for_one_year x lat x lon; if None, parameters are fitted from precipitation
    :param verbose: If True, print progress and summary statistics during calculation

    Outputs:
    :param spi: Standardized precipitation index (np.ndarray of shape time x lat x lon), or (alpha, beta) if save_parameters is True
    '''

    T, I, J = precipitation.shape

    # precipitation = precip*1000 # Convert precipitation from m to mm. Standard precip should be >> 1e-5 now.

    # Individual months and days are needed for later comparisons
    years = np.array([date.year for date in dates])
    months = np.array([date.month for date in dates])
    days = np.array([date.day for date in dates])

    N = int(np.ceil(T/len(np.unique(years)))) # Number of observations per year; round up for leap years

    # Initialize the spi values
    spi = np.ones((T, I ,J)) * np.nan

    # Reshape everything into 2D arrays for the easier calculations (fewer embedded loops).
    precipitation = precipitation.reshape(T, I*J)
    spi = spi.reshape(T, I*J)

    # Create one year of data with a leap year
    if T <= 366:
        one_year = np.array([datetime(2012, 1, 1) + timedelta(days = day) for day in range(366)])
    else:
        one_year = np.array([datetime(2012, 1, 1) + timedelta(days = day) for day in range(N)])

    if distribution_parameters is None:
        alpha = np.ones((one_year[:].size, I, J)) * np.nan
        beta = np.ones((one_year[:].size, I, J)) * np.nan
        alpha = alpha.reshape(one_year[:].size, I*J)
        beta = beta.reshape(one_year[:].size, I*J)
    else:
        alpha = distribution_parameters[0]; beta = distribution_parameters[1]
        alpha = alpha.reshape(alpha.shape[0], I*J)
        beta = beta.reshape(beta.shape[0], I*J)

    for t, date in tqdm(enumerate(one_year[:]), desc = 'Calculating SPI', disable = np.invert(verbose)):
        ind = np.where( (months == date.month) & (days == date.day) )[0]

        if len(ind) < 1:
            continue

        if distribution_parameters is None:
            # Precipitation is dsitribution according to a gamma distribution. Find the parameters of the gamme distribution
            A = np.log(np.nanmean(precipitation[ind,:]+1e-4, axis = 0)) - np.nansum(np.log(precipitation[ind,:]+1e-4), axis = 0)/N
            alpha[t,...] = (1/(4*A)) * (1 + np.sqrt(1+4*A/3))
            beta[t,...] = np.nanmean(precipitation[ind,:], axis = 0)/alpha[t,...]

        if save_parameters:
            continue

        # for ij in range(I*J):
        # Transform the values into a normal distribution with mean 0 and standard deviation 1 using the inverse CDF method.
        # That is, if P has a gamma distribution, then cdf(P) is a uniform distribution. Then cdf_n^-1(cdf(P)) is normally distributed.
            
        # cdf_n^-1 is the inverse of the normal cdf
        # the ppf, percent point function, is the inverse cdf. Its default arguements are mean (loc) = 0, and std (scale) = 1

        cdf = special.gammainc(alpha[t,:], precipitation[ind,:]/beta[t,:])

        # q = len(np.where(precipitation[ind,:] < 0.01)[0])/T # Get the weight of the number of points with precipitation at (or very close to) 0
        q = np.nansum(precipitation[ind,:] < 0.01, axis = 0)/T 
        cdf = q + (1-q)*cdf # Ajdust the cdf to account for the precip = 0 grid points

        # Points where the entire cdf is 0s and 1 is biasing the data. Had set the SPI NaN to remove the bias.
        #### This might be a class imbalance: 
        #.   Locations where no precip heavily outnumbers precip, the cdfs give no precip a probability of 1 and 0 to everything else.
        #.   This gives errors for pdf (prob 0 corresponds to -inf on the pdf, and +inf when prob is 1, or unrealistically small/large values when padding
        #.   is applied.
        #### Come back later to fix this class imbalance
        # if len(np.where( (cdf > 0.999) | (cdf < 0.001) )) == T:
            # spi[ind,:] = np.nan
            # continue

        # "Pad" the extreme cdf probability values (near 1 or 0) to prevent the calculated distribution from returning +/- inf
        cdf = np.where((cdf > 0.999), cdf-0.001, cdf)
        cdf = np.where((cdf < 0.001), cdf+0.001, cdf)

        spi[ind,:] = stats.norm.ppf(cdf, loc = 0, scale = 1)

    if save_parameters:
        return alpha.reshape(one_year[:].size, I, J), beta.reshape(one_year[:].size, I, J)

    # Return SPI to a 3D format
    spi = spi.reshape(T, I, J)

    return spi

def calculate_spei(
    precipitation, 
    pet, 
    dates, 
    D = None, 
    save_parameters = False, 
    distribution_parameters = None, 
    verbose = True
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    '''
    Calculate the standardized precipitation evapotranspiration index (SPEI) from precipitation and PET.

    Full details on SPEI can be found in Vicente-Serrano et al. 2010: https://doi.org/10.1175/2009JCLI2909.1.

    Inputs:
    :param precipitation: Daily precipitation dataset (np.ndarray of shape time x lat x lon)
    :param pet: Daily potential evapotranspiration dataset (np.ndarray of shape time x lat x lon)
    :param dates: Datetime labels for each timestep in precipitation and pet (np.ndarray of shape time)
    :param D: Pre-computed moisture deficit (P - PET) array of shape time x lat x lon; if None, computed as precipitation - pet
    :param save_parameters: If True, return fitted log-logistic distribution parameters instead of SPEI values
    :param distribution_parameters: Pre-computed parameters as (beta, alpha, gamma) arrays of shape time_for_one_year x lat x lon; if None, parameters are fitted from D
    :param verbose: If True, print progress messages during calculation

    Outputs:
    :param spei: Standardized precipitation evapotranspiration index (np.ndarray of shape time x lat x lon), or (beta, alpha, gamma) if save_parameters is True
    '''

    # Determine the difference between P and PET
    if D is not None:
        D = D
    else:
        D =  precipitation - pet

    T, I, J = D.shape

    # Individual months and days are needed for later comparisons
    years = np.array([date.year for date in dates])
    months = np.array([date.month for date in dates])
    days = np.array([date.day for date in dates])

    N = int(np.ceil(T/len(np.unique(years)))) # Number of observations per year; round up for leap years
    N_obs = len(np.unique(years)) # Number of of observations per time series

    # Define the constants given in Vicente-Serrano et al. 2010
    C0 = 2.515517
    C1 = 0.802853
    C2 = 0.010328

    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    # Create one year of data with a leap year
    if T <= 366:
        one_year = np.array([datetime(2012, 1, 1) + timedelta(days = day) for day in range(366)])
    else:
        one_year = np.array([datetime(2012, 1, 1) + timedelta(days = day) for day in range(N)])

    if distribution_parameters is None:
        # Initialize some variables
        frequencies = np.ones((T, I, J), dtype = np.float32) * np.nan
        PWM0 = np.ones((N, I, J), dtype = np.float32) * np.nan # Probability weighted moment of 0
        PWM1 = np.ones((N, I, J), dtype = np.float32) * np.nan # Probability weighted moment of 1
        PWM2 = np.ones((N, I, J), dtype = np.float32) * np.nan # Probability weighted moment of 2

        # Determine the frequency estimator and moments according eq. 3 in Vicente-Serrano et al. 2010
        if verbose:
            print('Calculating moments')
        for t, date in enumerate(dates[:N]):
            ind = np.where( (months == date.month) & (days == date.day) )[0]

            # Get the freqiemcy estimator
            frequencies[ind,:,:] = (stats.mstats.rankdata(D[ind,:,:], axis = 0) - 0.35)/N_obs

            # Get the moments
            PWM0[t,:,:] = np.nansum(((1 - frequencies[ind,:,:])**0)*D[ind,:,:], axis = 0)/N_obs
            PWM1[t,:,:] = np.nansum(((1 - frequencies[ind,:,:])**1)*D[ind,:,:], axis = 0)/N_obs
            PWM2[t,:,:] = np.nansum(((1 - frequencies[ind,:,:])**2)*D[ind,:,:], axis = 0)/N_obs

        # Calculate the parameters of the log-logistic distribution, using the equations in Vicente-Serrano et al. 2010
        if verbose:
            print('Calculating Pearson Type III distribution parameters')
        beta = (2 * PWM1 - PWM0)/(6*PWM1 - PWM0 - 6*PWM2) # Scale parameter

        alpha = (PWM0 - 2 * PWM1) * beta/(special.gamma(1 + 1/beta) * special.gamma(1 - 1/beta)) # Shape parameter

        gamma = PWM0 - alpha * special.gamma(1 + 1/beta) * special.gamma(1 - 1/beta) # Origin parameter

    else:
        alpha = distribution_parameters[0]; beta = distribution_parameters[1]; gamma = distribution_parameters[2]

    if save_parameters:
        return beta, alpha, gamma

    # Obtain the cumulative distribution of the moisture deficit
    if verbose:
        print('Calculating the CDF of P - PET')
    F = np.ones((T, I, J), dtype = np.float32) * np.nan

    for n, date in enumerate(one_year):
        ind = np.where( (date.month == months) & (date.day == days) )[0]

        if len(ind) < 1:
            continue
        else:
            for t in ind:
                F[t,:,:] = (1 + (alpha[n,:,:]/(D[t,:,:] - gamma[n,:,:]))**beta[n,:,:])**-1

    # Remove large variables as they are no longer needed
    if distribution_parameters is None:
        del frequencies, PWM0, PWM1, PWM2, beta, alpha, gamma
        gc.collect()

    # Calculate probabilities to convert to the standardized index
    prob = 1 - F
    prob = np.where(prob <= 0, 1e-5, prob) # Remove small probabilities of 0
    prob = np.where(prob >= 1, 1-1e-5, prob) # Remove large probabilities of 1

    # Initialize the SPEI
    spei = np.ones((T, I, J), np.float32) * np.nan

    # Reshape probability into a 2D array
    prob = prob.reshape(T, I*J)
    spei = spei.reshape(T, I*J)

    # Calculate the SPEI based on the inverse normal approximation given in Vicente-Serrano et al. 2010, Sec. 3
    if verbose:
        print('Converting probabilities to SPEI')
    
    prob_placeholder = prob.copy()
    # Determine whether to use prob or 1 - prob
    prob = np.where(prob <= 0.5, prob, 1 - prob)
    W = np.sqrt(-2 * np.log(prob))
    spei = (W - (C0 + C1*W + C2*(W**2))/(1 + d1*W + d2*(W**2) + d3*(W**3)))
    spei = np.where(prob_placeholder <= 0.5, spei, -1*spei)
    # SPEI_sign = 1
    # for ij in tqdm(range(I*J), desc = 'Calculating SPEI', disable = np.invert(verbose)):
    #     for t in range(T):

    #         # Determine whether to multiply the equation by 1 or -1, and whether to use prob or 1 - prob
    #         if prob[t,ij] <= 0.5:
    #             prob[t,ij] = prob[t,ij]
    #             SPEI_sign = 1
    #         else:
    #             prob[t,ij] = 1 - prob[t,ij]
    #             SPEI_sign = -1

    #         # Determine W
    #         W = np.sqrt(-2 * np.log(prob[t,ij]))

    #         # Calculate the normal distribution
    #         spei[t,ij] = SPEI_sign * (W - (C0 + C1*W + C2*(W**2))/(1 + d1*W + d2*(W**2) + d3*(W**3)))

    # Reshape SPEI back to a 3D array
    spei = spei.reshape(T, I, J)

    return spei
