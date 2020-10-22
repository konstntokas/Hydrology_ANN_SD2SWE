#Konstantin Ntokas
#
#
# 2019-11-17
# -----------------------------------------------------------------------------
# This function computes the logarithmic (or ignorance) score. Predictive distributions can
# be considered as Gaussian, Gamma distributed, Empirical or "Loi des fuites"
# (a Gamma distribution + a Dirac at zero, suitable for daily precip), and Kernel distribution.
#
# input: 
#           calculation:    mxn matrix; m = number of simulations  
#                                       n = number of member in ensemble 
#           observation:    mx1 vector; m = number of records
#           case:           - 'Normal'
#                           - 'Gamma'
#                           - 'Kernel'
#                           - 'Fuites'  is made for daily precipitation exclusively 
#                           - 'Empirical'
#           thres:          probability density threshold below which we consider that the
#                           event was missed by the forecasting system. This value must be
#                           small (e.g.: 0.0001 means that f(obs) given the forecasts is 
#                           only 0.0001 --> not forecasted).
#                           By default, thres = 0 and the logarithmic score is unbounded.
#          opt_case         - if 'case' = 'Fuites', opt_cas is the threshold to determine data
#                             which contributed to gamma distribution and those who are part of the
#                             Dirac impulsion
#                           - if 'case' = 'empirical', opt_cas needed is the number of bins
#                             in which to divide the ensemble, by default, it will be the
#                             number of members (Nan excluded). opt_cas have to be an integer
#                             superior to 1.
#
# output:
#          loga:            the logarithmic score (n*1 matrix)
#          ind_miss:        Boleans to point out days for which the event was missed according
#                           to the threshold specified by the user (1= missed) (n*1 matrix)
#
# Reference:
#          'Empirical' case is based on Roulston and Smith (2002) with
#          modifications -> quantile and members with similar values
# -----------------------------------------------------------------------------
# History
#
# MAB June 19: Added 2 cases for the empirical distribution: the
# observation can either be the smallest or the largest member of the
# augmented ensemble, in which case we can't use the "DeltaX = X(S+1) -
# X(S-1);" equation.
# -----------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm, gamma, gaussian_kde
import sys

def score_log(calculation, observation, case, thres=0., opt_case=None):
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
        
    # preparation 
    n = np.size(calculation, axis=0)
    loga = np.empty(n)
    loga[:] = np.nan
    ind_miss = np.empty(n)
    ind_miss[:] = np.nan
    
    # test input arguments are correct
    if len(observation) != n:
        sys.exit('Error! The length of the record of observations doesn''t match the length of the forecasting period')
    if thres == 0:
        print('Logarithmic score is unbounded')
    elif (thres < 0) or (thres > 1):
        sys.exit('Threshold has to be between 0 and 1.')
    
    # calcuation depending on the case
    if case == 'Empirical':
        # if no opt_case is given, number of bins are determined by the number of nonNaN members
        if opt_case == None:
            print('Bins used for empirical method determined by ensemble members')
        elif (opt_case < 2) or (not isinstance(opt_case, int)):
            sys.exit('Format of opt_case is not valide.')
        
        if  not isinstance(thres, float):
            sys.exit('Format of threshold is not valide. thres needs to be a list with 2 entries, determining the upper and lower bound for aberrant values')

        # loop over the records
        for j in range(n):
            # determine of observation is in the bound of max min of ensemble 
            if (~np.all(np.isnan(calculation[j,:]))) and (~np.isnan(observation[j])):
                if (np.nanmin(calculation[j,:]) <= observation[j]) and (observation[j] <= np.nanmax(calculation[j,:])):
                    ind_miss[j] = 0
                    # suppress NaN from the ensemble to determine the number of members
                    sample_nonnan = calculation[j,:][~np.isnan(calculation[j,:])]
                    sort_sample_nonnan = np.sort(sample_nonnan)
                    
                    # transform data, if bins are specified by user in the opt_case argument 
                    if opt_case != None:
                        sort_sample_nonnan = np.quantile(sort_sample_nonnan, np.arange(0, 1, 1/opt_case))
                    
                    # number of bins 
                    N = len(sort_sample_nonnan) 
                    
                    # if all members of forcast and obervation are the same -> perfect forecast
                    if len(np.unique(np.append(sort_sample_nonnan, observation[j]))) == 1:
                        proba_obs = 1
                    else:
                        # if some members are equal, modify slightly the value
                        if len(np.unique(sort_sample_nonnan)) != len(sort_sample_nonnan):
                            uni_sample = np.unique(sort_sample_nonnan)
                            bins = np.append(uni_sample, np.inf)
                            hist, binedges = np.histogram(sort_sample_nonnan, bins)
                            idxs, = np.where(hist > 1)
                            new_sample = uni_sample
                            for idx in idxs:
                                new_val = uni_sample[idx] + 0.01 *  np.random.rand(hist[idx]-1)
                                new_sample = np.append(new_sample, new_val)
                            sort_sample_nonnan = np.sort(new_sample)
                        # find position of the observation in the ensemble  
                        X = np.sort(np.concatenate((sort_sample_nonnan, observation[j])))
                        S, = np.where(X == observation[j])
                        # if observation is at the first or last position of the ensemble -> threshold prob
                        if S[0] == len(X)-1: 
                            proba_obs = thres
                        elif S[0] == 0: 
                            proba_obs = thres
                        else:
                            #if the observation falls between two members or occupies the first or last rank
                            if len(S) == 1:
                                # If the observation is between the augmented ensemble bounds
                                DeltaX = X[S[0]+1] - X[S[0]-1]
                                proba_obs = min(1/(DeltaX * (N+1)),1)
                            # if observation is equal to one member, choose the maximum of the probability density associated
                            elif len(S) == 2:
                                if S[0] == 0:
                                    DeltaX = X[S[1]+1] - X[S[1]]
                                elif S[1] == len(X)-1:
                                    DeltaX = X[S[0]] - X[S[0]-1]
                                else:
                                    DeltaX1 = X[S[1]+1] - X[S[1]]
                                    DeltaX2 = X[S[0]] - X[S[0]-1]
                                    DeltaX = min(DeltaX1,DeltaX2)
                                proba_obs = min(1/(DeltaX * (N+1)),1)
                            # test if probability below threshold
                            if proba_obs < thres:
                                proba_obs = thres
                                ind_miss[j] = 1
                # if observation is outside of the bound of the ensemble             
                else:
                    ind_miss[j] = 1
                    proba_obs = thres
                
                # calculate the logarithmus 
                loga[j] = - np.log2(proba_obs)
            # if all values are nan in ensemble   
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan
            
    elif case == 'Normal':
        if (opt_case != None):
            sys.exit('No optional case possible for Normal distribution')
        for j in range(n):
            # filter non nan values 
            sample_nonnan = calculation[j,:][~np.isnan(calculation[j,:])]
            # if there are values in the ensemble which are not nan
            if (len(sample_nonnan) > 0) and (~np.isnan(observation[j])):
                # perfect forecast, all member values equal the observation
                if len(np.unique(np.append(sample_nonnan, observation[j]))) == 1:
                    proba_obs = 1
                    ind_miss[j] = 0
                    loga[j] = - np.log2(proba_obs)
                else:
                    mu, sig = norm.fit(sample_nonnan)
                    # transform standard deviation to unbiased estimation of standard deviation
                    nb_mb = len(sample_nonnan)
                    sighat = nb_mb/(nb_mb-1) * sig
                    # all member forecasts the same but unequal the observation
                    if sighat == 0:
                        loga[j] = - np.log2(thres)
                        ind_miss[j] = 1
                    else:
                        proba_obs = min(norm.pdf(observation[j], mu, sighat), 1)
                        if proba_obs >= thres:
                            ind_miss[j] = 0
                            loga[j] = - np.log2(proba_obs)
                        else:
                            loga[j] = - np.log2(thres)
                            ind_miss[j] = 1
            # if all values in the snemble are nan      
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan
                
    elif case == 'Gamma':
        if (opt_case != None):
            sys.exit('No optional case possible for Gamma distribution')
        # check if any value is smaller equal zero
        idxs = np.where(calculation <= 0)
        if len(idxs[0]) == 0:
            for j in range(n):
                # filter non nan values 
                sample_nonnan = calculation[j,:][~np.isnan(calculation[j,:])]
                # if there are values in the ensemble which are not nan
                if (len(sample_nonnan) > 0) and (~np.isnan(observation[j])):
                    if len(np.unique(np.append(sample_nonnan, observation[j]))) == 1:
                        proba_obs = 1
                        ind_miss[j] = 0
                        loga[j] = - np.log2(proba_obs)
                    else:
                        # fit data to gamma distribtion 
                        alpha, loc, beta = gamma.fit(sample_nonnan, floc=0)
                        proba_obs = min(gamma.pdf(observation[j], alpha, loc, beta), 1)
                        if (alpha <= 0) or (beta <= 0):
                            loga[j] = - np.log2(thres)
                            ind_miss[j] = 1
                        else:
                            if proba_obs >= thres:
                                ind_miss[j] = 0
                                loga[j] = - np.log2(proba_obs)
                            else:
                                loga[j] = - np.log2(thres)
                                ind_miss[j] = 1
                # if all values in the snemble are nan      
                else:
                    loga[j] = np.nan
                    ind_miss[j] = np.nan
                
        else:
            sys.exit('Forecasts contain zeros. You must choose a different distribution.')

    elif case == 'Kernel':
        if (opt_case != None):
            sys.exit('No optional case possible for Kernel distribution')
            
        for j in range(n):
            # filter non nan values 
            sample_nonnan = calculation[j,:][~np.isnan(calculation[j,:])]
            # if there are values in the ensemble which are not nan
            if (len(sample_nonnan) > 0) and (~np.isnan(observation[j])):
                # perfect forecast, all member values equal the observation
                if len(np.unique(np.append(sample_nonnan, observation[j]))) == 1:
                    proba_obs = 1
                    ind_miss[j] = 0
                    loga[j] = - np.log2(proba_obs)
                else:
                    # all member forecasts the same but unequal the observation
                    if len(np.unique(sample_nonnan)) == 1:
                        loga[j] = - np.log2(thres)
                        ind_miss[j] = 1
                    else:
                        pd = gaussian_kde(sample_nonnan)
                        proba_obs = min(pd.pdf(observation[j]),1)
                        if proba_obs >= thres:
                            ind_miss[j] = 0
                            loga[j] = - np.log2(proba_obs)
                        else:
                            loga[j] = - np.log2(thres)
                            ind_miss[j] = 1
            # if all values in the snemble are nan      
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan
                   
    elif case == 'Fuites':
        if opt_case == None:
            sys.exit('Option missing for ''Fuites'' distribution.')
            
        for j in range(n):
            # filter non nan values 
            sample_nonnan = calculation[j,:][~np.isnan(calculation[j,:])]
            # if there are values in the ensemble which are not nan
            if (len(sample_nonnan) > 0) and (~np.isnan(observation[j])):
                # perfect forecast, all member values equal the observation
                if len(np.unique(np.append(sample_nonnan, observation[j]))) == 1:
                    proba_obs = 1
                    ind_miss[j] = 0
                    loga[j] = - np.log2(proba_obs)
                else:
                    idx_non_null, = np.where(sample_nonnan > opt_case)
                    prop_null = (len(sample_nonnan) - len(idx_non_null)) / len(sample_nonnan)
                    if observation[j] <= opt_case:
                        proba_obs = prop_null
                    else:
                        ens_non_null = sample_nonnan[idx_non_null]
                        # all member values above treshold equal, but unequal to observation
                        if len(np.unique(ens_non_null)) == 1:
                            proba_obs = thres
                        else:
                            # Fitting gamma parameters (max. likelihood method))
                            alpha, loc, beta = gamma.fit(ens_non_null, floc=0)
                            obs_val = gamma.pdf(observation[j], alpha, loc, beta) * (1-prop_null)
                            proba_obs = min(obs_val, 1)
                    # check if probability is above treshold
                    if proba_obs > thres:
                        loga[j] = - np.log2(proba_obs)
                        ind_miss[j] = 0
                    else:
                        loga[j] = - np.log2(thres)
                        ind_miss[j] = 1
            # if all values in the snemble are nan      
            else:
                loga[j] = np.nan
                ind_miss[j] = np.nan
            
    else:
        sys.exit('Choice of distribution type in ''cas'' is incorrect. Possible options are : "Normal", "Gamma", "Kernel", "Empirical" or "Fuites" ')
    
    S_LOG = np.nanmean(loga)
    ind_miss = np.nansum(ind_miss) 

    return S_LOG, ind_miss
