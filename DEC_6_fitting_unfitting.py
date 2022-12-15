import numpy as np; import pandas as pd
import scipy as sp; import scipy.stats as st
import torch; import torch.nn as nn; print(f"using torch version {torch.__version__}")
#use numba's just-in-time compiler to speed things up
from numba import njit
from sklearn.preprocessing import StandardScaler; from sklearn.model_selection import train_test_split
import matplotlib as mp; import matplotlib.pyplot as plt; 
#reset matplotlib stle/parameters
import matplotlib as mpl
#reset matplotlib parameters to their defaults
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('seaborn-deep')
mp.rcParams['agg.path.chunksize'] = 10000
font_legend = 15; font_axes=15
# %matplotlib inline
import copy; import sys; import os
from IPython.display import Image, display
from importlib import import_module

def main():
    import argparse
    import time
    # import sympy as sy
    import ipywidgets as wid; 
    
    IQN_BASE='/home/ali/Desktop/Pulled_Github_Repositories/torchQN'
    DATA_DIR='/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data'  
    print('BASE directoy properly set = ', IQN_BASE)
    utils_dir = os.path.join(IQN_BASE, 'utils')
    sys.path.append(utils_dir)
    import utils
    #usually its not recommended to import everything from a module, but we know
    #whats in it so its fine
    # from utils import *


        ################################### SET DATA CONFIGURATIONS ###################################
    X       = ['genDatapT', 'genDataeta', 'genDataphi', 'genDatam', 'tau']

    FIELDS  = {'RecoDatam' : {'inputs': X, 
                            'xlabel':  r'$m$ (GeV)', 
                            'xmin': 0, 
                            'xmax': 25},
            
            'RecoDatapT': {'inputs': ['RecoDatam']+X, 
                            'xlabel':  r'$p_T$ (GeV)' , 
                            'xmin'  : 20, 
                            'xmax'  :  80},
            
            'RecoDataeta': {'inputs': ['RecoDatam','RecoDatapT'] + X, 
                            'xlabel': r'$\eta$',
                            'xmin'  : -5,
                            'xmax'  :  5},
            
            'RecoDataphi'  : {'inputs': ['RecoDatam', 'RecodatapT', 'RecoDataeta']+X,
                            'xlabel': r'$\phi$' ,
                            'xmin'  : -3.2, 
                            'xmax'  :3.2}
            }
    
    all_variable_cols=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']
    all_cols=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam', 'tau']
    ################################### Load unscaled dataframes ###################################
    SUBSAMPLE=int(1e3)#subsample use for development - in production use whole dataset
    train_data=pd.read_csv(os.path.join(DATA_DIR,'train_data_10M_2.csv'),
                        usecols=all_cols,
                        nrows=SUBSAMPLE
                        )

    test_data=pd.read_csv(os.path.join(DATA_DIR,'test_data_10M_2.csv'),
                        usecols=all_cols,
                        nrows=SUBSAMPLE
                        )


utils.explore_data(df=train_data, title='Unscaled Dataframe')
if __name__ == '__main__':
    main()