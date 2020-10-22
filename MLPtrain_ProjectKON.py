# Description: 
# An ensemble of 'nb_members' MLPs is trained for 'nb_epochs'. Hereby, 11 inputs
# are taken. (Same inputs as in the paper except snow density from ERA5) Each MLP consists
# of a single hidden layer with 'nb_hid' neurons. The validation data set is simulated
# by using the trained networks. The result on the validation data set is evaluated. 
# Caution, between each application the IPython console needs to be closed because 
# of the tensorflow saver constructor, called in MLPtrain. Otherwise networks are
# not saved correctly. (There most be a work around, but could not find any, sorry!)
# -----------------------------------------------------------------------------

import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from MLPtrain import MLP
from performance import performance
import os
import shutil

# -----------------------------------------------------------------------------
# TABLE OF CONTENT
#
# 0. CREATE CONDA ENVIROMENT TO INSTALL USED PACKAGES 
# 1. TRAINING OF MLPS
# 2. EVALUATION ON VALIDATION DATA SET
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 0.  CREATE CONDA ENVIROMENT TO INSTALL USED PACKAGES 

# ## create conda enviroment 
# conda create -n SD2SWE python=3.6.10
# conda activate SD2SWE 

# ##Add packages 
# conda install numpy=1.16.4 pandas geopandas matplotlib tensorflow scikit-learn ipykernel 

# ## install spyder in enviroment 
# conda install spyder

# ## open spyder
# spyder
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Data requirement 
# 1. ALready prepared data set for training and validation

    
# Input
# 1. train, validation
# 	- 2 dim DataFrame saved by pickle library
# 	- index: station ID
# 	- columns: valiables (see Jupyter notebook)
# 3. save_path: assign path where to save networks and results; new folder will be created
# 5. nb_members: determines the number of memebers in the ensemble 
# 6. nb_hid: determines the number of neurons in the hidden layer
# 7. nb_epochs: dertermins the number of epochs in training

# Output:
#   - saves trained networks of MLPs and scaler for later use to save_path/saved_MLPs
#   - estimation SWE and evaluation of estimation on validation data set;
#     results saved to save_path/results
# -----------------------------------------------------------------------------


def MLPtrain_workflow(train, val, save_path, nb_members, nb_hid, nb_epochs):
    
    # generate folder for saved MLP networks and results 
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    save_path_results = '{0}/results'.format(save_path)
    os.makedirs(save_path_results)
    save_path_MLPs = '{0}/saved_MLPs'.format(save_path)
    os.makedirs(save_path_MLPs)
    
    # -----------------------------------------------------------------------------
    # 1. TRAINING OF MLPS
    
    # determine variables used for MLP optimisation 
    variables = ['SD', 'SWE', 'Day of year', 'days without snow', 'number frost-defrost',
                 'accum pos degrees', 'average age SC', 'number layer', 'accum solid precip',
                 'accum solid precip in last 10 days', 'total precip last 10 days', 'average temp last 6 days']
    MLP_train = train[variables]
    # delete all rows with nan values 
    MLP_train = MLP_train.dropna()
    # select explanatory and target variables 
    y_train = MLP_train['SWE'].values.astype('float32') 
    x_train = MLP_train.drop('SWE', axis=1).values.astype('float32')
        
    # determination of MLP setup 
    activ_fc = 'tanh'
    init_w = 2
    init_b = 2
    optAlg = 'Adadelta'
    batch_size = 100
    shuf_data = 1
    
    # optimise MLP ensmeble by function; trained MLP networks and scaler for standardisation are saved to save_path
    MLP(x_train, y_train, nb_epochs, nb_members, nb_hid, save_path_MLPs,
        activ_fc, init_w, init_b, optAlg, batch_size, shuf_data)
    del train
    
    # -----------------------------------------------------------------------------
    # 2. EVALUATION ON VALIDATION DATA SET
    
    # Perturbe SD on validation data set and estimate SWE; evaluate estimated SWE 
    # against observation of SWE
    # load scaler for standardisation         
    [scalerIn, scalerOut] = pickle.load(open(save_path_MLPs + '/scaler', "rb"))
    
    # determine variables used for MLP optimisation 
    variables = ['SD', 'SWE', 'Day of year', 'days without snow', 'number frost-defrost',
                 'accum pos degrees', 'average age SC', 'number layer', 'accum solid precip',
                 'accum solid precip in last 10 days', 'total precip last 10 days', 'average temp last 6 days']
    MLP_val = val[variables]
    # delete all rows with nan values 
    MLP_val = MLP_val.dropna()
    # select explanatory and target variables 
    y_val = MLP_val['SWE'].values.astype('float32')    
    x_val = MLP_val.drop('SWE', axis=1).values.astype('float32')
    # find index for SD for perturbation 
    idx_SD = MLP_val.drop('SWE', axis=1).columns.get_loc('SD')
    
    # initialise matrix for ensemble with 400 members
    ensemble400 = np.empty((len(y_val), 20*20))
    
    # assign model setup 
    nb_members = 20 
    
    for mb in range(nb_members):
        # create network graph
        tf.reset_default_graph()
        imported_graph = tf.train.import_meta_graph(save_path_MLPs +  "/mb_{0}.ckpt.meta".format(mb))
        with tf.Session() as sess:
            # restore parameter
            imported_graph.restore(sess, save_path_MLPs +  "/mb_{0}.ckpt".format(mb))
            
            # get prediction with noisy inputs as an ensemble 
            for k in range(x_val.shape[0]):
                line_input = x_val[k, :]
                input_net = np.tile(line_input, (20, 1))
                SD = line_input[idx_SD]
                if SD < 20: 
                    SD_low = SD - 1
                    SD_high = SD + 1
                else: 
                    SD_low = SD * 0.95
                    SD_high = SD * 1.05
                SD_noise_1rec = np.random.uniform(low=SD_low, high=SD_high, size=20)
                input_net[:, idx_SD] = SD_noise_1rec
                input_net_std = scalerIn.transform(input_net)
                predict_std = sess.run("op_to_restore:0", feed_dict={"input:0": input_net_std}).flatten()
                ensemble400[k, mb*20:(mb+1)*20] = scalerOut.inverse_transform(predict_std).flatten()   
    
    # determine 20 quantiles, to get 20 members 
    ensemble20 = np.quantile(ensemble400, np.arange(0.025, 1, 1/20), axis=1).transpose()
    
    # save ensemble          
    pickle.dump(ensemble20, open(save_path_results + '/ensembleVal_SD_pt', "wb"))   

    # evaluate on validation data set            
    # apply performance function, saves graphics and results (as csv) in folder assigned above
    performance(ensemble20, y_val, save_path_results)
        
    print(datetime.now())
    print('Evaluation on validation data set done')