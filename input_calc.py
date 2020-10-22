# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:12:57 2019

@author: ntok1701
"""

import pandas as pd 
import numpy as np
from datetime import datetime, date
import pickle


def get_met_data(Lat, Long, Station, year, par_dir):
    # get lat and long of each Nivo Station rounded on one digit
    Lat_Meteo = np.round(Lat, 1)
    Long_Meteo = np.round(Long, 1)
    
    # set up dates for meteo data
    start = datetime(year, 1, 1)
    if year == 2019:
        end = datetime(year, 4, 30)
    else:
        end = datetime(year, 12, 31)
    dates_Meteo = pd.date_range(start, end)
    
    # get precip, tmin, tmax for each Nivo Station for each day 
    num_station = len(Lat)
    num_dates = len(dates_Meteo)
    total_precip = np.empty((num_station, num_dates))
    total_precip[:] = np.nan
    tmax = np.empty((num_station, num_dates))
    tmax[:] = np.nan
    tmin = np.empty((num_station, num_dates))
    tmin[:] = np.nan
    print(year)
    print(datetime.now())
    if year == 2019:
        months = [1, 2, 3, 4]
    else:
        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for month in months:
        data_tp = pickle.load(open(par_dir + "/00_data/saved_variables/ERA5/ERA5_data_tp_{0:02d}_{1:04d}".format(month, year), "rb"))  
        data_tmintmax = pickle.load(open(par_dir + "/00_data/saved_variables/ERA5/ERA5_data_tmaxtmin_{0:02d}_{1:04d}".format(month, year), "rb"))  
        date_start = datetime(year, month, 1)
        num_days = len(data_tp['Date'])
        idx_date, = np.where(dates_Meteo == np.datetime64(date_start))
        data_tp['Lat'] = np.round(data_tp['Lat'], decimals=1)
        data_tp['Long'] = np.round(data_tp['Long'], decimals=1)
        for i in range(num_station): 
            idx_lat = np.where(data_tp['Lat'] == Lat_Meteo[i])
            idx_lon = np.where(data_tp['Long'] == Long_Meteo[i])
            total_precip[i, int(idx_date):int(idx_date+num_days)] = np.squeeze(data_tp['TotalPrecip(mm)'][: ,idx_lat, idx_lon])
            tmin[i, int(idx_date):int(idx_date+num_days)] = np.squeeze(data_tmintmax['Min_t2m(C)'][: ,idx_lat, idx_lon])
            tmax[i, int(idx_date):int(idx_date+num_days)] = np.squeeze(data_tmintmax['Max_t2m(C)'][: ,idx_lat, idx_lon])
            
    dates_Meteo = dates_Meteo.values.astype('datetime64[D]')
    
    return total_precip, tmin, tmax, dates_Meteo

def frost_defrost(tmin_mod, tmax_mod, dates, nb_dates):
    Frost_NoDefrost = np.nonzero(np.logical_and(tmin_mod<=-1, tmax_mod<1).values)
    Frost_Defrost = np.nonzero(np.logical_and(tmin_mod<=-1, tmax_mod>=1).values)
    NoFrost_NoDefrost = np.nonzero(np.logical_and(tmin_mod>-1, tmax_mod<1).values)
    NoFrost_Defrost = np.nonzero(np.logical_and(tmin_mod>-1, tmax_mod>=1).values)

    episode = np.empty(nb_dates)
    episode[Frost_NoDefrost] = 1
    episode[Frost_Defrost] = 2
    episode[NoFrost_NoDefrost] = 3
    episode[NoFrost_Defrost] = 4
    count = 0 
    frost = 0
    result = np.zeros(nb_dates)
    for i in range(nb_dates):
        date = dates[i]
        if (date.month == 9) and (date.day == 1):
            count = 0
        if episode[i] == 1: 
            frost = 1
        elif episode[i] == 2: 
            count += 1
            frost = 0
        elif episode[i] == 3: 
            count = count 
            frost = frost
        elif episode[i] == 4: 
            if frost == 1:
                count += 1
                frost = 0
        result[i] = count 
    result_df = pd.DataFrame(result, index=dates)
    return result_df

def num_without_snow(tmax_mod, total_precip_mod, dates, nb_dates):
    result = np.zeros(nb_dates)
    logic = ~np.logical_and(tmax_mod<=0, total_precip_mod>=3)
    count = 0
    for i in range(nb_dates):
        date = dates[i]
        if (date.month == 9) and (date.day == 1):
            count = 0
        if logic[i] == 1:
            count += 1
        result[i] = count
    result_df = pd.DataFrame(result, index=dates)   
    return result_df

def pos_degrees(tmid, dates, nb_dates):
    result = np.zeros(nb_dates)
    tmid_mod = np.where(tmid > 0, tmid, 0)
    count = 0
    for i in range(nb_dates):
        date = dates[i]
        if (date.month == 9) and (date.day == 1):
            count = 0
        count = count + tmid_mod[i]
        result[i] = count
    result_df = pd.DataFrame(result, index=dates)   
    return result_df

def num_layer(delta, th, total_precip_solid, dates, nb_dates):
    result = np.zeros(nb_dates)
    count = 0
    new_layer = 0
    cumul = np.sum(total_precip_solid[:delta])
    if cumul > th: 
        count += 1
        new_layer = 1
        result[:delta] = 1
    for i in range(delta, nb_dates):
        date = dates[i]
        if (date.month == 9) and (date.day == 1):
            count = 0
        if (new_layer == 0) and (total_precip_solid[i] > 0):
            count += 1
            new_layer = 1
        cumul = np.sum(total_precip_solid[(i-delta) + 1 :i + 1])
        if (new_layer == 1) and (cumul <= th):
            new_layer = 0
        result[i] = count 
    result_df = pd.DataFrame(result, index=dates)   
    return result_df

def age_snow_cover(d, tmid, total_precip_solid):
    month = d.month
    year = d.year
    if month < 9:
        year = year - 1
    else: 
        year = year
    sep_1st = date(year, 9, 1)
    
    tmid_mod = tmid.loc[sep_1st:d]
    nb = len(tmid_mod)
    total_precip_mod_solid = total_precip_solid[sep_1st:d]
    cuml = np.sum(total_precip_mod_solid)
    age = np.arange(nb, 0, -1)
    age_acc = np.sum(age * total_precip_mod_solid)
    result = age_acc/cuml
    return result, total_precip_mod_solid, cuml, nb
