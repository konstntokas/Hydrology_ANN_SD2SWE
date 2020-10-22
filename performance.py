import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from PerformanceFunctions import CRPS, DecompCRPS, Reliability, ScoreLog, TalagrandRank

def performance(ensemble, obs, filename_fig):

    # calculate the MAE, RMSE and MBE of median 
    # calculate the median
    median = np.nanmedian(ensemble, axis=1)
    # calculate MAE, MBE and RSME between median and validation
    MAE = np.nanmean(np.absolute(median - obs)) 
    RMSE = np.sqrt(np.nanmean((median - obs) ** 2))
    MBE = np.nanmean(median - obs)
    
    # histogram of simulated medians and the validation dataset
    max_median = median.max()
    max_obs = obs.max()
    max_all = np.array([max_median, max_obs]).max()
    bin_nb = int(max_all/40)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist([median, obs],
             bins=bin_nb,
             color=['red', 'blue'],
             label=['median simulation', 'observation'])
    ax.set_xlabel('value of SWE in $mm$')
    ax.set_ylabel('count of sim/obs')
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    # save figure
    strFile = filename_fig + '/histogram.png'
    if os.path.isfile(strFile):
        os.remove(strFile)
    fig.savefig(strFile)
    plt.close()
    
    # calculate the CRPS by using the empirical case
    CRPS_emp = CRPS.CRPS(ensemble, obs, 'emp')
    
    # decompose the CRPS 
    total, reliability, potential = DecompCRPS.decomp_CRPS(ensemble, obs)
    
    # get reliability diagram 
    nomi, Meff, Mlen = Reliability.Reliability(ensemble, obs)
    # creat diagram and save it 
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(nomi, Meff, color='black', marker='o', linestyle='None')
    ax.plot([0,1], [0,1], color='red',  linestyle='dashed')
    ax.set_xlabel('forecast probabilities')
    ax.set_ylabel('observed relat. freq.')
    plt.grid(True)
    plt.tight_layout()
    
    # save figure
    strFile = filename_fig + '/reliabilitiy_diagramm.png'
    if os.path.isfile(strFile):
        os.remove(strFile)
    fig.savefig(strFile)
    plt.close()
    
    # get the log/ignorance score
    S_LOG_emp, ind_miss_emp = ScoreLog.score_log(ensemble, obs, 'Empirical', thres=0.001)
    
    # calculate the Talagrand Rank histogramm
    hist, ranks = TalagrandRank.Talagrand_rank(ensemble, obs)
    # creat diagram and save it 
    x = np.arange(len(hist))
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.bar(x, hist)
    ax.set_xticks([0, 4, 9, 14, 20])
    ax.set_xticklabels([1, 5, 10, 15, 21])
    ax.set_xlabel('rank of observation')
    ax.set_ylabel('observed freq.')
    plt.tight_layout()

    # save figure
    strFile = filename_fig + '/Talagrand_diagramm.png'
    if os.path.isfile(strFile):
        os.remove(strFile)
    fig.savefig(strFile)
    plt.close()

    # save results to csv file
    results = np.array([MAE, RMSE, MBE, CRPS_emp, reliability, potential, S_LOG_emp])
    results = np.round(results, 2)
    results_pd = pd.DataFrame(results, index=['MAE', 'RMSE', 'MBE', 'CRPS emp',
                                              'reliability', 'potential', 'Ignorance score emp'])
    results_pd.to_csv(filename_fig + '/results.csv', index=True)
    
    print('performance analysis finished')

