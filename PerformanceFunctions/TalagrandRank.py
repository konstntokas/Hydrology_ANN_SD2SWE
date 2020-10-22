# Konstantin Ntokas
#
#
# 2019-11-18
# -----------------------------------------------------------------------------
# This function calcultes the input for rank histgram/ Talagrand diagramm. It 
# can be used to determine if the distribution of the forecast is biased.
#
# input: 
#           calculation:    mxn matrix; m = number of simulations  
#                                       n = number of member in ensemble 
#           observation:    mx1 vector; m = number of records   
#
# output:   
#           hist:           1x(n+1) vactor; frequency of each bin;
#                                           nb bins = nb members + 1 obs  = nb ranks
#           ranks:          mx1 vector; indicates the rank for each ensemble sample 
# -----------------------------------------------------------------------------
import numpy as np


def Talagrand_rank(calculation, observation):
    # transform input into numpy array 
    calculation = np.array(calculation, dtype='float64')
    observation = np.array(observation, dtype='float64')
    dim1 = calculation.shape
    if len(dim1) == 1: 
        calculation = calculation.reshape((1,dim1[0]))
    dim2 = observation.shape
    if len(dim2) == 0: 
        observation = observation.reshape((1,1))
    elif len(dim2) == 1:
        observation = observation.reshape((dim2[0],1))
        
    #initialisation
    m, n = calculation.shape
    ranks = np.empty(m)
    ranks[:] =  np.nan

    # loop over each ensemle sample     
    for i in range(m):
        if ~np.isnan(observation[i]):
            if np.all(~np.isnan(calculation[i,:])):
                calc_obs = np.append(calculation[i,:], observation[i])
                calc_obs_sort = np.sort(calc_obs)
                idxs, = np.where(calc_obs_sort == observation[i])
                if len(idxs) > 1:  
                    rand_idx = int(np.floor(np.random.rand(1) * len(idxs)))
                    ranks[i] = idxs[rand_idx]
                else: 
                    ranks[i] = idxs[0]   
    
    # find the frequency of each bin 
    ranks_nonnan = ranks[~np.isnan(ranks)]
    bins = np.arange(-0.5, n+1.5, 1)
    hist, bin_edges = np.histogram(ranks_nonnan, bins=bins)

    return hist, ranks

