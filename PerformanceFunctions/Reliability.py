#Konstantin Ntokas
#
#
# 2019-11-17
# -----------------------------------------------------------------------------
# This function calcultes the input data for the reliability diagramm
#
# input: 
#           calculation:    mxn matrix; m = number of simulations  
#                                       n = number of member in ensemble 
#           observation:    mx1 vector; m = number of records   
#
# output:   
#          nomi:            nominal probability of the intervals
#          Meff:            mean effective probability of the intervals
#          Mlen:            mean effective length of the intervals
# -----------------------------------------------------------------------------
import numpy as np


def Reliability(calculation, observation):
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


    # Nominal probability of the intervals
    bin_start = np.arange(0.05, 0.5, 0.05)
    bin_end = np.arange(0.95, 0.5, -0.05)
    bins = np.concatenate((bin_start, bin_end))
    nb = len(bin_start)
    nomi = bin_end - bin_start
    
    # initialisation
    L = np.size(calculation, axis=0)
    length = np.zeros((L,nb))
    eff = np.zeros((L,nb))
    
    for i in range(L):
        if ~np.isnan(observation[i]):
            # get quantile for each bin and the median
            q = np.quantile(calculation[i,:], bins)
            qmed = np.median(calculation[i,:])
                
            # Compute lenghts of intervals (for Luc Perreault)
            length[i,:] = q[nb:] - q[:nb]
            
            # Locate observation in the ensemble
            if observation[i] <= qmed:
                eff[i,:] = q[:nb] <= observation[i]
            else:
                eff[i,:] = q[nb:] >= observation[i]
        else: 
            length[i, :] = np.nan
            eff[i, :] = np.nan
        
    # Compute averages
    Meff = np.nanmean(eff, axis=0)
    Mlen = np.nanmean(length, axis=0)  # (For Luc)

    return nomi, Meff, Mlen
