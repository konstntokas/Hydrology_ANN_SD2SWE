#Konstantin Ntokas
#
#
# 2019-11-17
# -----------------------------------------------------------------------------
# This function calcultes the decomposition of CRPS according to Hersbach (2000)
#
# input: 
#           calculation:    mxn matrix; m = number of simulations  
#                                       n = number of member in ensemble 
#           observation:    mx1 vector; m = number of records   
#
# output:   
#           total:          uncertainty + potential 
#           uncertainty:    represents the uncertainty of the CRPS
#           potential:      represents the potential CRPS for a perfectly certain system
# -----------------------------------------------------------------------------
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('error')


def decomp_CRPS(calculation, observation):
    # preparation 
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

    m, n = calculation.shape
    alpha = np.zeros((m,n+1))
    beta = np.zeros((m,n+1))
    
    for i in range(m):
        # if observation does not exist, no calculation for alpha and beta
        if ~np.isnan(observation[i]):
            ensemble_sort = np.sort(calculation[i]);
            for k in range(n+1):
                if k == 0:
                    if observation[i] < ensemble_sort[0]:
                        alpha[i,k] = 0
                        beta[i,k] = ensemble_sort[0] - observation[i]
                    else:
                        alpha[i,k] = 0
                        beta[i,k] = 0
                elif k == n:
                    if observation[i] > ensemble_sort[n-1]:
                        alpha[i,k] = observation[i] - ensemble_sort[n-1]
                        beta[i,k] = 0
                    else:
                        alpha[i,k] = 0
                        beta[i,k] = 0
                else:
                    if observation[i] > ensemble_sort[k]:
                        alpha[i,k] = ensemble_sort[k] - ensemble_sort[k-1]
                        beta[i,k] = 0
                    elif observation[i] < ensemble_sort[k-1]:
                        alpha[i,k] = 0
                        beta[i,k] = ensemble_sort[k] - ensemble_sort[k-1]
                    elif (observation[i] >= ensemble_sort[k-1]) and (observation[i] <= ensemble_sort[k]):
                        alpha[i,k] = observation[i] - ensemble_sort[k-1]
                        beta[i,k] = ensemble_sort[k] - observation[i]
                    else:
                        alpha[i,k] = np.nan
                        beta[i,k] = np.nan
        else: 
            alpha[i,:] = np.nan
            beta[i,:] = np.nan
            
            
    alpha1 = np.nanmean(alpha, axis=0)
    beta1 = np.nanmean(beta, axis=0)
    
    g = alpha1 + beta1
    o = beta1 / g
    
    weight = np.arange(n+1) / n 
    uncertainty = np.nansum(g * np.power(o - weight, 2))
    potential = np.nansum(g * o * (1 - o))
    total = uncertainty + potential
                    
    return total, uncertainty, potential