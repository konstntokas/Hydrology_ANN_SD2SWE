# Hydrology_ANN_SD2SWE
## Codes for the Article 'Investigating ANN architectures and training to estimate SWE from snow depth'
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4276414.svg)](https://doi.org/10.5281/zenodo.4276414)

A Jupyter Notebook (Documentation.ipynb) is provided in this folder as well, which coveres the whole project workflow step by step, including:
1. Create conda enviroment to install used packages
1. Preparation of data
1. Training of single MLP model
1. Evaluation of training
1. Multiple MLP model
1. Simulation and evaluation on unseen data set

Furthermore, 00_Overview.pdf depicts the relationships between the python scripts in the main folder to train the MLP ensemble. The main script is MLPtrain_ProjectKON_main.py. In the folder 00_saved_variables are some small sample data sets for execution to facilitate the understanding of the codes.

The input/output data set in .txt format for the MLP training, testing and validation can be obtained [here](https://can01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdataverse.harvard.edu%2Fdataset.xhtml%3FpersistentId%3Ddoi%3A10.7910%2FDVN%2FT46ANR&amp;data=04%7C01%7Cmarie-amelie.boucher%40usherbrooke.ca%7C15c6928147524d7954c608d87c1df1af%7C3a5a8744593545f99423b32c3a5de082%7C0%7C0%7C637395815355934392%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C1000&amp;sdata=QqAFel6%2B6KxaVr6fM53mLH0qXg06BhubaoXsfc%2FJIyI%3D&amp;reserved=0). This data does not include used data from partners, which do not allow to publish their data. However, 98% of the used data can be handed out. 
