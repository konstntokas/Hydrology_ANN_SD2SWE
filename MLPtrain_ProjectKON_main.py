# This script can be used to train the MLPs of the project of Konstantin 
# Caution, between each application the IPython console needs to be closed because 
# of the tensorflow saver constructor, called in MLPtrain. Otherwise networks are
# not saved correctly. (There most be a work around, but could not find any, sorry!)
# -----------------------------------------------------------------------------

import pickle 
from MLPtrain_ProjectKON import MLPtrain_workflow

# -----------------------------------------------------------------------------
# TABLE OF CONTENT
#
# 0. CREATE CONDA ENVIROMENT TO INSTALL USED PACKAGES 
# 1. LOAD MLP INPUT DATA 
# 2. DETERMINE NETWORK STRUCTURE AND PATH TO SAVE
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 0.  CREATE CONDA ENVIROMENT TO INSTALL USED PACKAGES 

# ## create conda enviroment 
# conda create -n SD2SWE python=3.6.10
# conda activate SD2SWE 

# ##Add packages 
# conda install numpy=1.16.4 pandas geopandas  matplotlib tensorflow scikit-learn ipykernel 

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

# assign snow class 
# List of snow classes: ['Canada', 'Ephemeral snow', 'Prairie snow', 'Tundra snow',
#                        'Taiga snow', 'Mountain snow', 'Maritime snow']
sn_class = 'Ephemeral snow'

# -----------------------------------------------------------------------------
# 1 LOAD MLP INPUT DATA 
train = pickle.load(open("00_saved_variables/data_train_perturbed", "rb"))
val = pickle.load(open("00_saved_variables/data_val", "rb"))

# cut data into snow class
if sn_class == 'Canada': 
    train_SC = train
    val_SC = val
else:
    # get data for certain snow class
    train_SC = train.loc[train['snowclass'] == sn_class]
    val_SC = val.loc[val['snowclass'] == sn_class]
# -----------------------------------------------------------------------------
# 2. DETERMINE NETWORK STRUCTURE AND PATH TO SAVE

# look up table for network structure of SingleMLP and MultipleMLP 
# (optimsed MLPs by Konstantin Ntokas)
# 
# 1. singleMLP: one ensemble of MLP over entire Canada
#           - nb_hid:    120
#           - nb_epochs: 5
# 2. MultipleMLP: one ensemble of MLPs for each snow class; snow classes are
#                 defined by Sturm et al. (2009) https://doi.org/10.5065/D69G5JX5
#                       Ephem.  Prair.  Tund.  Taig.  Mount.  Marit. 
#           - nb_hid:     12     24      48     96     192     192
#           - nb_epochs:  50     30      30     20      20      20


# determine path to save, new folder will be created
save_path = '01_savedMLP_ProjectKON/{}'.format(sn_class)
nb_members = 20
nb_hid = 12
nb_epochs = 50

MLPtrain_workflow(train_SC, val_SC, save_path, nb_members, nb_hid, nb_epochs)