# Alireza Amani started, Konstantin Ntokas finished
#
#
# 2019-11-16
# -----------------------------------------------------------------------------
# This function calcultes the CRPS between an ensemble and one observation. 
#
# input: 
#           calculation:    mxn matrix; m = number of simulations  
#                                       n = number of member in ensemble 
#           observation:    mx1 vector; m = number of records   
#           case:           'emp':              empirical cumulative distribution
#                           'normal_exact':     normal cumulative distribution
#                           'gamma_exact':      gamma cumulative distribution
#
# output:   
#           CRPS:           mx1 vactor; each record is one entry 
# -----------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm, gamma 

 
def CRPS(calculation, observation, case):
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

    # initilisation      
    m = np.size(calculation, axis=0)
    CRPS = np.empty((m, 1))
    CRPS.fill(np.nan)
    
    # non-parametric estimation based on the empirical cumulative distribution of the ensemble. According to Luc Perreault's idea
    if (case == "emp"):
        for i in range(m):
            if (np.any(np.isnan(calculation[i,:])) == 0 and np.isnan(observation[i]) == 0):
                ssample = np.sort(calculation[i,:])
                step_size = 1/(len(calculation[i,:]))
                
                # caluculation of the area below the observation
                area1 = 0
                sub_sample1 = ssample[ssample <= observation[i]]
                sub_sample1 = np.append(sub_sample1, observation[i])
                for j in range(1,len(sub_sample1)):
                    area1 += (j*step_size)**2 * (sub_sample1[j] - sub_sample1[j-1])

                # caluculation of the area above the observation
                area2 = 0
                sub_sample2 = ssample[ssample > observation[i]]
                sub_sample2 = np.insert(sub_sample2, 0, observation[i])
                n2 = len(sub_sample2)
                for j in range(1,n2):
                    area2 += ((n2-j)*step_size)**2 * (sub_sample2[j] - sub_sample2[j-1])
                    
                CRPS[i] = area1 + area2
                
            else:
                CRPS[i] = np.nan
                
    # -------------------------------------------------------------------------
    # estimation based on the normal cumulative distribution of the ensemble               
    elif (case == "normal_exact"):
        for i in range(m):
            if (np.any(np.isnan(calculation[i,:])) == 0 and np.isnan(observation[i]) == 0):
                # preparation
                mu, sigma = norm.fit(calculation[i,:])
                # transform standard deviation to unbiased estimation of standard deviation
                nb_mb = len(calculation[i,:])
                sighat = nb_mb/(nb_mb-1) * sigma
                vcr = (observation[i] - mu) / sighat
                phi = norm.pdf(vcr,  loc=0, scale=1)
                PHI = norm.cdf(vcr,  loc=0, scale=1)
                # calculation of the CRPS according to Gneiting and Raftery 2007
                CRPS[i] = abs(sighat * ((1/np.sqrt(np.pi)) - 2*phi - (vcr*(2*PHI-1))))
            else: 
                CRPS[i] = np.nan
            
    # -------------------------------------------------------------------------
    # estimation based on the gamma cumulative distribution of the ensemble   
    elif (case == "gamma_exact"):
        for i in range(m):
            if (np.any(np.isnan(calculation[i,:])) == 0 and np.isnan(observation[i]) == 0):
                # preparation; exchange negative values in the data
                sample = calculation[i,:]
                idxs, = np.where(sample <= 0)
                for idx in idxs: 
                    sample[idx] = 0.0001
                    
                # fit data to gamma distribtion 
                alpha, loc, beta = gamma.fit(sample, floc=0)
                # generate cumulative gamma distribution
                data1 = gamma.rvs(alpha, loc=0, scale=beta, size=1000)  
                data2 = gamma.rvs(alpha, loc=0, scale=beta, size=1000)  
                CRPS[i]= np.mean(np.absolute(data1 - observation[i])) - 0.5 * np.mean(np.absolute(data1 - data2))
            else: 
                CRPS[i] = np.nan

    return np.nanmean(CRPS)     
            
        

    
