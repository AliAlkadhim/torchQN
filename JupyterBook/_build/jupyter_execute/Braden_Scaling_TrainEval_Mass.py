#!/usr/bin/env python
# coding: utf-8

# # Braden-Scaling IQNx4: 1.Mass (UNDER CONSTRUCTION!)

# # Do `source setup.sh` before trying to run this notebook!
# 
# 

# ## External Imports
# 
# If you don't have some of these packages installed, you can also use the conda environment that has all of the packages by doing `conda env create -f IQN_env.yml && conda activate IQN_env`
# 
# There is also a `requirements.txt` here so that it can be run on an interactive website, eg binder or people can `pip install` it.

# In[1]:


import numpy as np; import pandas as pd
import scipy as sp; import scipy.stats as st
import torch; import torch.nn as nn; print(f"using torch version {torch.__version__}")
#use numba's just-in-time compiler to speed things up
from numba import njit
from sklearn.preprocessing import StandardScaler; from sklearn.model_selection import train_test_split
import matplotlib as mp; print('matplotlib version= ', mp.__version__)

import matplotlib.pyplot as plt; 
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

try:
    import optuna
    print(f"using (optional) optuna version {optuna.__version__}")
except Exception:
    print('optuna is only used for hyperparameter tuning, not critical!')
    pass
import argparse
import time
# import sympy as sy
import ipywidgets as wid; 


# ## Import utils, and set environemnt variables

# need to tune latest braden scaling hyperparameters on cluster.
# 
# see if i can find/write a decorator to add the current cell to a file which will be run on cluster. I think %writefile file.py could work, by adding the cell to another file... 
# 

# In[2]:


# 'IQN' in 
# some_environment={}
# some_environment.update(os.environ())
# some_environment
# 'DATA' in list(os.environ)


# In[3]:


# os.environ['IQN_BASE']='/home/ali/Desktop/Pulled_Github_Repositories/torchQN'
# os.environ['DATA_DIR']='/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/data'


# ### A user is competent enought to do `source setup.sh` on a `setup.sh` script that comes in the repo, such as the next cell uncommented

# In[52]:


#!/bin/bash
export IQN_BASE= $pwd #/home/ali/Desktop/Pulled_Github_Repositories/torchQN

#DAVIDSON
export DATA_DIR='/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/data'
#LOCAL
export DATA_DIR='/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data'
echo DATA DIR
ls -l $DATA_DIR
#ln -s $DATA_DIR $IQN_BASE, if you want
#conda create env -n torch_env -f torch_env.yml
conda activate torch_env
mkdir -p ${IQN_BASE}/images/loss_plots ${IQN_BASE}/trained_models  ${IQN_BASE}/hyperparameters ${IQN_BASE}/predicted_data
tree $IQN_BASE


# In[4]:


# env = {}
# env.update(os.environ)
# env.update(source(os.environ["IQN_BASE"])) 

try:
    IQN_BASE = os.environ['IQN_BASE']
    print('BASE directoy properly set = ', IQN_BASE)
    utils_dir = os.path.join(IQN_BASE, 'utils')
    sys.path.append(utils_dir)
    import utils
    #usually its not recommended to import everything from a module, but we know
    #whats in it so its fine
    from utils import *
    print('DATA directory also properly set, in %s' % os.environ['DATA_DIR'])
except Exception:
    # IQN_BASE=os.getcwd()
    print("""\nBASE directory not properly set. Read repo README.\
    If you need a function from utils, use the decorator below, or add utils to sys.path\n
    You can also do 
    os.environ['IQN_BASE']=<ABSOLUTE PATH FOR THE IQN REPO>
    or
    os.environ['IQN_BASE']=os.getcwd()""")
    pass


# # 2. Helper Functions
# 
# Plotting and image functions

# In[5]:


def show_jupyter_image(image_filename, width = 1300, height = 300):
    """Show a saved image directly in jupyter. Make sure image_filename is in your IQN_BASE !"""
    display(Image(os.path.join(IQN_BASE,image_filename), width = width, height = height  ))
    
    
def use_svg_display():
    """Use the svg format to display a plot in Jupyter (better quality)"""
    from matplotlib_inline import backend_inline
    backend_inline.set_matplotlib_formats('svg')

def reset_plt_params():
    """reset matplotlib parameters - often useful"""
    use_svg_display()
    mpl.rcParams.update(mpl.rcParamsDefault)
    


def show_plot(legend=False):
    use_svg_display()
    plt.tight_layout();
    plt.show()
    if legend:
        plt.legend(loc='best')
        
def set_figsize(get_axes=False,figsize=(7, 7)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    if get_axes:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        return fig, ax
    
def set_axes(ax, xlabel, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None, title=None):
    """saves a lot of time in explicitly difining each axis, its title and labels: do them all in one go"""
    use_svg_display()
    ax.set_xlabel(xlabel,fontsize=font_axes)
    if ylabel:
        ax.set_ylabel(ylabel,fontsize=font_axes)
    if xmin and xmax:
        ax.set_xlim(xmin, xmax)
    
    if ax.get_title()  != '':
        #if the axes (plot) does have a title (which is non-empty string), display it 
        ax.set_title(title)
    if ax.legend():
        #if an axis has a legned label, display it
        ax.legend(loc='best',fontsize=font_legend)
    if ymin and ymax:
        #sometimes we dont have ylimits since we do a lot of histograms, but if an axis has ylimits, set them
        ax.set_ylim(ymin, ymax)
    
    try:
        fig.show()
    except Exception:
        pass
    plt.tight_layout()
    # plt.show()

def get_finite(values):
    return values[np.isfinite(values)]

def mkdir(dir_):
    """make a directory without overwriting what's in it if it exists"""
    # assert isinstance(dir_, str)
    try:
        os.system('mkdir -p %s' % str(dir_) )
    except Exception:
        pass
    
############################ Some decorators ############################ 
def SourceIQN(func):
    def _func(*args):
        import os
        from common.utility.source import source
        env = {}
        env.update(os.environ)
        env.update(source(os.environ["IQN_BASE"]))        
        func(*args, env=env)
    return _func


def time_type_of_func(tuning_or_training, _func=None):
    def timer(func):
        """Print the runtime of the decorated function"""
        import functools
        import time
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            if tuning_or_training=='training':
                print(f'training IQN to estimate {target}')
            elif tuning_or_training=='tuning':
                print(f'tuning IQN hyperparameters to estimate {target}')
            start_time = time.perf_counter()    
            value = func(*args, **kwargs)
            end_time = time.perf_counter()      
            run_time = end_time - start_time    
            if tuning_or_training=='training':
                print(f"training target {target} using {func.__name__!r} in {run_time:.4f} secs")
            elif tuning_or_training=='tuning':
                print(f"tuning IQN hyperparameters for {target} using {func.__name__!r} in {run_time:.4f} secs")
            return value
        return wrapper_timer
    if _func is None:
        return timer
    else:
        return timer(_func)

def debug(func):
    """Print the function signature and return value"""
    import functools
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  
        signature = ", ".join(args_repr + kwargs_repr)           
        print(f"Calling {func.__name__}({signature})")
        values = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {values!r}")           
        return values
    return wrapper_debug


def make_interactive(func):
    """ make the plot interactive"""
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        plt.ion()
        output=func(*args, **kwargs)
        plt.ioff()
        return output
    return wrapper
        
from IPython.core.magic import register_cell_magic

@register_cell_magic
def write_and_run(line, cell):
    """write the current cell to a file (or append it with -a argument) as well as execute it
    use with %%write_and_run at the top of a given cell"""
    argz = line.split()
    file = argz[-1]
    mode = 'w'
    if len(argz) == 2 and argz[0] == '-a':
        mode = 'a'
    with open(file, mode) as f:
        f.write(cell)
    get_ipython().run_cell(cell)
    


# <a name="Results_prior"></a>
#  
#  
# # Results prior to Braden-scaling
# 
# 
# 
# Recall that the best IQNx4 autoregressive results that I attained prior to trying the Braden scaling was the following (which was implemented in the Davidson cluster here: `/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/OCT_7/*.py` and copied to my repo [here](https://github.com/AliAlkadhim/torchQN/tree/master/OCT_7) )
# 

# In[6]:


show_jupyter_image('OCT_7/AUTOREGRESSIVE_RESULTS_OCT7.png',width = 800, height = 200)


# So we know IQNx4 works (but not perfect enough), but in this notebook we try the Braden scaling (first time trying this scaling) to see if we can do better.

# In[7]:


# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=True)

# set a seed to ensure reproducibility
seed = 128
rnd  = np.random.RandomState(seed)
#sometimes jupyter doesnt initialize MathJax automatically for latex, so do this:
wid.HTMLMath('$\LaTeX$')


# ## Set arguments and configurations

# In[8]:


# add_to_cluster()
################################### ARGUMENTS ###################################
parser=argparse.ArgumentParser(description='train for different targets')
parser.add_argument('--N', type=str, help='''size of the dataset you want to use. 
                    Options are 10M and 100K and 10M_2, the default is 10M_2''', required=False,default='10M_2')
#N_epochs X N_train_examples = N_iterations X batch_size
# N_iterations = (N_epochs * train_data.shape[0])/batch_size
#N_iterations = (N_epochs * train_data.shape[0])/64 = 125000 for 1 epoch
parser.add_argument('--n_iterations', type=int, help='''The number of iterations for training, 
                    the default is''', required=False,default=50)
#default=5000000 )
parser.add_argument('--n_layers', type=int, help='''The number of layers in your NN, 
                    the default is 5''', required=False,default=6)
parser.add_argument('--n_hidden', type=int, help='''The number of hidden layers in your NN, 
                    the default is 5''', required=False,default=6)
parser.add_argument('--starting_learning_rate', type=float, help='''Starting learning rate, 
                    the defulat is 10^-3''', required=False,default=1.e-2)
parser.add_argument('--show_loss_plots', type=bool, help='''Boolean to show the loss plots, 
                    default is False''', required=False,default=False)
parser.add_argument('--save_model', type=bool, help='''Boolean to save the trained model dictionary''', 
                    required=False,default=False)
parser.add_argument('--save_loss_plots', type=bool, help='''Boolean to save the loss plots''', 
                    required=False,default=False)


################################### CONFIGURATIONS ###################################
DATA_DIR=os.environ['DATA_DIR']
JUPYTER=True
use_subsample=False
if use_subsample:
    SUBSAMPLE=int(1e5)#subsample use for development - in production use whole dataset
else:
    SUBSAMPLE=None
    


# In[9]:


get_ipython().run_line_magic('pinfo', 'plt.rcsetup.interactive_bk')


# In[64]:


if JUPYTER:
    # print(plt.rcsetup.interactive_bk )
    # matplotlib interactive mode: ion or ioff
    plt.ioff()
    print('interactive? ', mpl.is_interactive())
    args = parser.parse_args(args=[])
    N = '10M_2'
    n_iterations = int(1e4)
    n_layers, n_hidden = int(1), int(10)
    starting_learning_rate = float(1.e-2)
    show_loss_plots = False
    save_model=False
    save_loss_plots = False
else:
    args = parser.parse_args()
    N = args.N
    n_iterations = args.n_iterations
    n_layers = args.n_layers
    n_hidden = args.n_hidden
    starting_learning_rate=args.starting_learning_rate
    show_loss_plots=args.show_loss_plots
    save_model=args.save_model
    save_loss_plots=args.save_loss_plots


# ### Import the numpy data, convert to dataframe and save (if you haven't saved the dataframes)

# # Explore the Dataframe and preprocess

# # Data

# In[65]:


use_svg_display()
show_jupyter_image('images/pythia_ppt_diagram.png', width=2000,height=500)


# In[66]:


###############################################################################################
y_label_dict ={'RecoDatapT':'$p(p_T)$'+' [ GeV'+'$^{-1} $'+']',
                    'RecoDataeta':'$p(\eta)$', 'RecoDataphi':'$p(\phi)$',
                    'RecoDatam':'$p(m)$'+' [ GeV'+'$^{-1} $'+']'}

loss_y_label_dict ={'RecoDatapT':'$p_T^{reco}$',
                    'RecoDataeta':'$\eta^{reco}$', 
                    'RecoDataphi':'$\phi^{reco}$',
                    'RecoDatam':'$m^{reco}$'}


# <!-- For Davidson team, please read try to all the code/comments before asking me questions! -->

# Decide on an evaluation order 

# In[67]:


################################### SET DATA CONFIGURATIONS ###################################
X       = ['genDatapT', 'genDataeta', 'genDataphi', 'genDatam', 'tau']

#set order of training:
#pT_first: pT->>m->eta->phi
#m_first: m->pT->eta->phi




ORDER='m_First'

if ORDER=='m_First':
    FIELDS  = {'RecoDatam' : {'inputs': X, 
                               'xlabel':  r'$m$ (GeV)', 
                              'ylabel':'$m^{reco}$',
                               'xmin': 0, 
                               'xmax': 25},
                           

               'RecoDatapT': {'inputs': ['RecoDatam']+X, 
                               'xlabel':  r'$p_T$ (GeV)' , 
                              'ylabel': '$p_T^{reco}$',
                               'xmin'  : 20, 
                               'xmax'  :  80},

               'RecoDataeta': {'inputs': ['RecoDatam','RecoDatapT'] + X, 
                               'xlabel': r'$\eta$',
                               'ylabel':'$\eta^{reco}$',
                               'xmin'  : -5,
                               'xmax'  :  5},

               'RecoDataphi'  : {'inputs': ['RecoDatam', 'RecodatapT', 'RecoDataeta']+X,
                               'xlabel': r'$\phi$' ,
                                'ylabel' :'$\phi^{reco}$',
                               'xmin'  : -3.2, 
                               'xmax'  :3.2}
              }


# Load and explore raw (unscaled) dataframes

# In[262]:


all_variable_cols=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']
all_cols=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam', 'tau']
################################### Load unscaled dataframes ###################################
print(f'SUBSAMPLE = {SUBSAMPLE}')
raw_train_data=pd.read_csv(os.path.join(DATA_DIR,'train_data_10M_2.csv'),
                      usecols=all_cols,
                      nrows=SUBSAMPLE
                      )

raw_test_data=pd.read_csv(os.path.join(DATA_DIR,'test_data_10M_2.csv'),
                      usecols=all_cols,
                     nrows=SUBSAMPLE
                     )


# In[263]:


def explore_data(df, title, scaled=False):
    fig, ax = plt.subplots(1,5, figsize=(15,10) )
    # df = df[['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']]
    levels = ['RecoData', 'genData']
    kinematics=['pT','eta','phi','m']
    columns = [level+k for level in levels for k in kinematics]
    print(columns)
    columns = columns + ['tau']
    print(columns)
    df = df[columns]
    
    for k_i, k in enumerate(kinematics):
        Reco_var = levels[0]+k
        gen_var = levels[1]+k
        print('Reco_var: ', Reco_var, ', \t gen_var: ', gen_var)
        ax[k_i].hist(df[Reco_var], bins=100, label=Reco_var, alpha=0.35)
        ax[k_i].hist(df[gen_var], bins=100, label=gen_var, alpha=0.35)
        xmin, xmax = FIELDS[Reco_var]['xmin'], FIELDS[Reco_var]['xmax']
        xlabel=FIELDS[Reco_var]['xlabel']
        ax[k_i].set_xlim( (xmin, xmax) )
        # set_axes(ax[k_i], xlabel=xlabel, ylabel='', xmin=xmin, xmax=xmax)
        ax[k_i].set_xlabel(xlabel,fontsize=26)
        
        
                  
        if scaled:
            ax[k_i].set_xlim(df[gen_var].min(),df[gen_var].max() )
        
        ax[k_i].legend(loc='best', fontsize=13)
    ax[4].hist(df['tau'],bins=100, label=r'$\tau$')
    ax[4].legend(loc='best', fontsize=13)
    fig.suptitle(title, fontsize=30)
    show_plot()


# In[264]:


explore_data(df=raw_train_data, title='Unscaled Dataframe')


# In[265]:


print(raw_train_data.shape)
raw_train_data.describe()#unscaled


# In[266]:


print(raw_test_data.shape)
raw_test_data.describe()#unscaled


# In[267]:


# np.array(train_data['genDatapT'])


# # Scaling
# 
# scaling (or standarization, normalization) is someimes done in the following way:
# $$ X' = \frac{X-X_{min}}{X_{max}-X_{min}} \qquad \rightarrow \qquad X= X' (X_{max}-X_{min}) + X_{min}$$

# In[268]:


# def standarize(values):
#     expected_min, expected_max = values.min(), values.max()
#     scale_factor = expected_max - expected_min
#     offset = expected_min
#     standarized_values = (values - offset)/scale_factor 
#     return standarized_values


# Or by taking z-score:
# 
# $$ X'=z(X)=\frac{X-E[X]}{\sigma_{X}}  \qquad \rightarrow \qquad X = z^{-1}(X')= X' \sigma_{X} + E[X].$$
# 

# -----------
# 
# ## Basically a "standard scaling procedure" is the following (background):
# 
# 1. Split the data into train and test dataframes
# 2. fit the scaler on each of the train and test sets independently, that is, get the mean and std, ( optionally and min and max or other quantities) of each feature (column) of each of the train and test dataframes, independently.
# 3. transform each of the train and test sets independently. That is, use the means and stds of each column to transform a column $X$ into a column $X'$ e.g. according to 
# $$ X'=z(X)= \frac{X-E[X]}{\sigma_{X}}$$
# 4. Train NN on transformed features $X_{train}'$ (and target $y_{train}'$) (in train df, but validate on test set, which will not influence the weights of NN ( just used for observation that it doesnt overfit) )
# 5. Once the training is done, *evaluate the NN on transformed features of the test set* $X_{test}'$, i.e. do $NN(X_{test}')$, which will result in a scaled prediction of the target $y_{pred}'$
# 6. Unscale the $y_{pred}'$, i.e. apply the inverse of the scaling operation, e.g.
# $$ y_{pred}=z^{-1}(y_{pred}')= y_{pred}' \sigma_{y} + E[y]$$,
# where 
# 
# $$\sigma_y$$
# 
# and 
# 
# $$E[y]$$
# 
# are attained from the test set *prior to training and scaling*.
# 
# 7. Compare to $y$ (the actual distribution you're trying to estimate) one-to-one

# In[269]:


use_svg_display()
show_jupyter_image('images/scaling_forNN.jpg', width=2000,height=500)


# # Braden scaling 
# 
# In the IQN-scipost overleaf, we say the scaling is the following:
# 
# $$\mathbb{T}(p_T) = z(\log p_T), \qquad \mathbb{T}(\eta) = z(\eta), \qquad \mathbb{T}(\phi) = z(\phi), \qquad \mathbb{T}(m) = z(\log (m + 2))$$ 
# 
# $$ \mathbb{T}(\tau) = 6\tau - 3 $$
# 
# 
# Which means, for a jet observable $\mathcal{O}$ (or quantile $\tau$), the Braden-scaling perscribes that the data is first scaled according to:
# 
# $$
# \begin{align}
#     \mathbb{L} (\mathcal{O} \mid \mathcal{O} \in X ) &=
#     \begin{cases}
#         z \left( \log{\mathcal{O}} \right), \qquad & \text{if } \mathcal{O}= p_T \\
#         z \left(\mathcal{O} \right), \qquad & \text{if } \mathcal{O}=\eta \\
#         z \left( \log (\mathcal{O} + 2) \right), \qquad & \text{if } \mathcal{O}=m \\
#         z \left( \mathcal{O} \right), \qquad & \text{if } \mathcal{O}=\phi \\
#         z \left( 6 \mathcal{O} -3 \right), \qquad & \text{if } \mathcal{O}=\tau
#     \end{cases}
# \end{align}
# $$
# 
# Note that the equation above describes the scaling for the training features $X$. The point of taking the log is 
# 
# The targets, as we say in the paper, are chosen to be the following:
# 
# $$
# z\left(\frac{y_n + 10}{x_n + 10}\right), \qquad n = 1,\cdots,4,
# \label{eq:normalization}
# $$
# 
# We mean that the predicted target for a desired reco observable $\mathcal{O}$ is chosen to be the following function
# 
# $$
#     \mathbb{T}(\mathcal{O} \mid \mathcal{O} \in y) = z \left( \frac{\mathbb{L} (\mathcal{O}^{\text{reco}}) +10 }{\mathbb{L}(\mathcal{O}^{\text{gen}}) +10} \right),
# $$
# 
# where for a random variable $x$, $z$ is the standardization function (z-score):
# 
# $$
#    x'= z (x) \equiv \frac{x-\bar{x}}{\sigma_{x}} \ .
# $$
# 
# Such that its inverse is 
# 
# $$ z^{-1}(x') = x' \ \sigma_x + \bar{x} $$
# 
# If we do this on the data, after training, the NN $f_{\text{IQN}}$ will not estimate the observable, 
# 
# $$\mathcal{O}^{\text{predicted}} \ne \mathcal{O}^{\text{reco}}$$
# 
# but will instead estimate 
# 
# $$
#         f_{\text{IQN}} (\mathcal{O}) \approx  z \left( \frac{\mathbb{L} (\mathcal{O}^{\text{reco}}) +10 }{\mathbb{L}(\mathcal{O}^{\text{gen}}) +10} \right),
# $$
# 
# which needs to be de-scaled (when evaluated on the data that which has been scaled according to 
# 
# $$\mathbb{T}(\text{evaluation data}) = z \left( \frac{\mathbb{L} (\text{data}^{\text{reco}}) +10 }{\mathbb{L}(\text{data}^{\text{gen}}) +10} \right) $$
# 
# 
# in order to copare with $\mathcal{O}$ directly.) The descaling for $\mathcal{O}=p_T$ (as an example) would be:
# 
# $$
#     p_T^{\text{predicted}} = \mathbb{L}^{-1} \left[ z^{-1} (f_{\text{IQN}} ) \left[ \mathbb{L} (p_T^\text{gen})+10 \right] -10 \right]
# $$
# 

# -------------
# 
# # Scale the data accoding to the "Braden Kronheim scaling" :
# 

# In[270]:


def z(x):
    eps=1e-20
    return (x - np.mean(x))/(np.std(x)+ eps)
def z_inverse(xprime, x):
    return xprime * np.std(x) + np.mean(x)


# In[271]:


def get_scaling_info(df):
    """args: df is train or eval df.
    returns: dictionary with mean of std of each feature (column) in the df"""
    features=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam',
              'RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam', 'tau']
    SCALE_DICT = dict.fromkeys(features)
    for i in range(8):
        feature = features[i]
        feature_values = np.array(df[feature])
        SCALE_DICT[feature]={}
        SCALE_DICT[feature]['mean'] = np.mean(feature_values)
        SCALE_DICT[feature]['std'] = np.std(feature_values)
    return SCALE_DICT


# In[272]:


TRAIN_SCALE_DICT = get_scaling_info(raw_train_data);print(TRAIN_SCALE_DICT)
print('\n\n')
TEST_SCALE_DICT = get_scaling_info(raw_test_data);print(TEST_SCALE_DICT)


# In[273]:


def L(orig_observable, label):
    eps=1e-20
    orig_observable=orig_observable+eps
    if label=='pT':
        const=0
        log_pT_=np.log(orig_observable) 
        L_observable = log_pT_
    if label=='eta':
        const=0
        L_observable=orig_observable
    if label=='m':
        const=2
        L_observable=np.log(orig_observable + const)
    if label=='phi':
        L_observable=orig_observable
    if label=='tau':
        L_observable=orig_observable
#         L_observable = (6*orig_observable) - 3
    
    return L_observable.to_numpy()


# In[274]:


def L_inverse(L_observable, label):
    eps=1e-20
    L_observable=L_observable+eps
    if label=='pT':
        const=0
        L_inverse_observable = np.exp(L_observable)
    if label=='eta':
        L_inverse_observable = L_observable
    if label=='m':
        const=2
        L_inverse_observable = np.exp(L_observable) - const
    if label=='tau':
        L_inverse_observable=L_observable
        # L_inverse_observable = (L_observable+3)/6
        
    if not isinstance(L_inverse_observable, np.ndarray):
        L_inverse_observable = L_inverse_observable.to_numpy()
    return L_inverse_observable


# $$
#     \mathbb{T}(\mathcal{O}) = z \left( \frac{\mathbb{L} (\mathcal{O}^{\text{reco}}) +10 }{\mathbb{L}(\mathcal{O}^{\text{gen}}) +10} \right),
# $$
# 

# In[275]:


def T(variable, scaled_df):
    if variable=='pT':
        L_pT_gen=scaled_df['genDatapT']
        L_pT_reco = scaled_df['RecoDatapT']
        target = (L_pT_reco+10)/(L_pT_gen+10) 
    if variable=='eta':
        L_eta_gen=scaled_df['genDataeta']
        L_eta_reco = scaled_df['RecoDataeta']
        target =  (L_eta_reco+10)/(L_eta_gen+10) 
    if variable=='phi':
        L_phi_gen=scaled_df['genDataphi']
        L_phi_reco = scaled_df['RecoDataphi']
        target =  (L_phi_reco+10)/(L_phi_gen+10) 
    if variable=='m':
        L_m_gen=scaled_df['genDatam']
        L_m_reco = scaled_df['RecoDatam']
        target =  (L_m_reco+10)/(L_m_gen+10) 
    
    return target


# In[276]:


def L_scale_df(df, title, save=False):
    #scale
    # SUBSAMPLE=int(1e6)
    df = df[all_cols]#[:SUBSAMPLE]
    # print(df.head())
    scaled_df = pd.DataFrame()
    #select the columns by index: 
    # 0:genDatapT, 1:genDataeta, 2:genDataphi, 3:genDatam, 
    # 4:RecoDatapT, 5:RecoDataeta, 6:RecoDataphi, 7: Recodatam
    scaled_df['genDatapT'] = L(df.iloc[:,0], label='pT')
    scaled_df['RecoDatapT'] = L(df.iloc[:,4], label='pT')
    
    scaled_df['genDataeta'] = L(df.iloc[:,1], label='eta')
    scaled_df['RecoDataeta'] = L(df.iloc[:,5],label='eta')
    
    
    scaled_df['genDataphi'] = L(df.iloc[:,2],label='phi')
    scaled_df['RecoDataphi'] = L(df.iloc[:,6],label='phi')

    scaled_df['genDatam'] = L(df.iloc[:,3],label='m')
    scaled_df['RecoDatam'] = L(df.iloc[:,7],label='m')
    #why scale tau?
    # scaled_df['tau'] = 6 * df.iloc[:,8] - 3
    scaled_df['tau'] = L(df.iloc[:,8],label='tau')
    
    print(scaled_df.describe())
    
    if save:
        scaled_df.to_csv(os.path.join(DATA_DIR, title) )
    return scaled_df


# ## If you want to generate the Scaled data frames, run the cell below

# In[277]:


scaled_train_data = L_scale_df(raw_train_data, title='scaled_train_data_10M_2.csv',
                             save=True)
print('\n\n')
scaled_test_data = L_scale_df(raw_test_data,  title='scaled_test_data_10M_2.csv',
                            save=True)

explore_data(df=scaled_train_data, title='Braden Kronheim-L-scaled Dataframe', scaled=True)


# ### If you want to load the previously generated scaled dataframe, run the cell below

# In[290]:


target = 'RecoDatam'
source  = FIELDS[target]
features= source['inputs']
########

print('USING NEW DATASET\n')
#UNSCALED
# train_data_m=pd.read_csv(os.path.join(DATA_DIR,'train_data_10M_2.csv'),
#                        usecols=features,
#                        nrows=SUBSAMPLE)

# print('TRAINING FEATURES\n', train_data.head())

# test_data_m= pd.read_csv(os.path.join(DATA_DIR,'test_data_10M_2.csv'),
#                        usecols=features,
#                        nrows=SUBSAMPLE)
# print('\nTESTING FEATURES\n', test_data.head())
# valid_data= pd.read_csv(os.path.join(DATA_DIR,'valid_data_10M_2.csv'),
#                        usecols=features,
#                        nrows=SUBSAMPLE)


# SCALED
train_data_m=pd.read_csv(os.path.join(DATA_DIR,'scaled_train_data_10M_2.csv'),
                       usecols=all_cols,
                       nrows=SUBSAMPLE)

print('TRAINING FEATURES\n', train_data_m.head())

test_data_m= pd.read_csv(os.path.join(DATA_DIR,'scaled_test_data_10M_2.csv'),
                       usecols=all_cols,
                       nrows=SUBSAMPLE)
print('\nTESTING FEATURES\n', test_data_m.head())

print('\ntrain set shape:',  train_data_m.shape)
print('\ntest set shape:  ', test_data_m.shape)
# print('validation set shape:', valid_data.shape)

scaled_train_data = train_data_m
scaled_test_data = test_data_m


# In[291]:


labels = ['pT', 'eta','phi','m']
fig, ax=plt.subplots(1,1)
for label in labels:
    target_ = T(label, scaled_df=scaled_train_data)
    
    ax.hist(target_, label = '$T($' +label+ '$)$', alpha=0.4 )
set_axes(ax=ax, xlabel='pre-z ratio targets T', )
plt.show()


# ---------
# ------
# # ML
# 
# Note that this ideas is very powerful and has the potential to replace the use of Delphes/GEANT for most people. According to the [previous paper](https://arxiv.org/pdf/2111.11415.pdf) this method already works for a single IQN, and we know it works reasonably well for autoregressive IQNx4, [as we said above](#Results_prior) .
# 
# It's important to remember "the master formula" of all of machine learning:
# 
# $$\int \frac{\partial L}{\partial f} p(y|x) dy  = 0 \tag{1}$$
# 
# or, equivalently, 
# 
# $$ \frac{\delta R}{\delta f}=0,$$
# 
# where $L$ is the loss function, $f$ is the model (neural network/classifier/regressor, etc. In this case it's an IQN) (implicitly parameterized by potentially a  gazillion parameters), 
# 
# and $p(y|x)$ the PDF of targets $y$ that we want to estimate, given (set of) features $x$, $R$ is the risk functional (sometime called objective function or cost function):
# 
# $$ R[f] = \int \cdots \int \, p(y, \mathbf{x}) \, L(f(\mathbf{x}, \theta), y) \, dy \, d\mathbf{x}$$
# 
# 
# where the $R[f]$ is approximated by the normalized sum of the losses over the all the samples. So, for IQNs,
# 
# $$ L_{\text{IQN}}(f, y)=\left\{\begin{array}{ll}
# \tau(y-f(\boldsymbol{x}, \tau ; \boldsymbol{\theta})) & y \geq f(\boldsymbol{x}, \tau ; \boldsymbol{\theta}) \\
# (1-\tau)(f(\boldsymbol{x}, \tau ; \boldsymbol{\theta})-y) & y<f(\boldsymbol{x}, \tau ; \boldsymbol{\theta})
# \end{array},\right.$$
# 
# Means that what was done previously is that the risk functional, which could be a functional of many models $f$, was a only a functional of a single model: $R[f_1,..., f_n] = R[f_1]$. Here we have 4 models 
# 
# $$R_{\text{IQN}x4} =R_{\text{IQN}}[f_m, f_{p_T}, f_\eta, f_\phi], $$ 
# 
# and since we're choosing the evaluation order:
# 
# $$
# \begin{align}
#     p(\mathbf{y} | \mathbf{x}) & = 
#     p(m'|\mathbf{x} )\nonumber\\
#     & \times p(p_T'|\mathbf{x}, m' )\nonumber\\
#     & \times p(\eta'| \mathbf{x}, m', p_T' )\nonumber\\
#       & \times p(\phi' |  \mathbf{x}, m', p_T', \eta' ) ,
# \end{align}
# $$
# 
# 
# 
# $$ \begin{align}
# R_{\text{IQN}x4} &= \int L_\text{IQN} \left( f_m (\mathbf{x_m},\tau), \mathbf{y_m} \right) p(\mathbf{x_m, y_m})  d \mathbf{x_m} d \mathbf{y_m} \\
# &\times \  ... \times \\ 
# &\times \int L_\text{IQN} \left( f_\phi (\mathbf{x_\phi},\tau), \mathbf{y_\phi} \right) p(\mathbf{x_\phi, y_\phi})  d \mathbf{x_\phi} d \mathbf{y_\phi}
# \end{align}$$
# 
# where, again, each model $f_i$ is also dependent on a set of parameters $\theta_i$ (dropped for simplicity).

# Our risk functional is minimized for
# 
# $$\frac{\delta R_{\text{IQN}x4} }{\delta f_m}=0\tag{5}$$
# 
# (which is basically what's done in the training process to get $f_m^{*}$ whose weights/parameters minimize the loss). Suppose we factorize the risk as
# 
# $$ R_{\text{IQN}x4}  = R_{\text{IQN}}^m \ R_{\text{IQN}}^{p_T}  \ R_{\text{IQN}}^\eta \ R_{\text{IQN}}^\phi \tag{6},$$ 
# 
# then, by Eq (4),
# 
# $$R_{\text{IQN}}^m \equiv \int L_\text{IQN} \left( f_m (\mathbf{x_m},\tau), \mathbf{y_m} \right) p(\mathbf{x_m, y_m,\tau})  d \mathbf{x_m} d \mathbf{y_m} d \mathbf{\tau},
# $$
# and by Eq (5)
# 
# $$\int d \mathbf{x_m} d \mathbf{y_m} d \mathbf{\tau} \ p(\mathbf{x_m, y_m,\tau})   \ \frac{ \delta L_\text{IQN} \left( f_m (\mathbf{x_m},\tau), \mathbf{y_m} \right) }{\delta f_m} = 0$$
# 
# 
# and by Eq (2)
# 
# $$
# \int d \mathbf{x_m} d \mathbf{y_m} d \mathbf{\tau} \ p(\mathbf{x_m, y_m,\tau})   \ \frac{ \delta L_\text{IQN} \left( f_m (\mathbf{x_m},\tau), \mathbf{y_m} \right) }{\delta f_m} = 0 \tag{7}
# $$
# 
# >> ...
# <br>
# 
# Expand Eq (2) in Eq (7) and integrate wrt y over the appropriate limits to see that  $f(\mathbf{x},\mathbf{\tau})$ is the quantile function for $p(\mathbf{y}|\mathbf{x})$, i.e. (I believe) that IQNx4 should work basically exactly.

# $$R_{\text{IQN}x4} = [ L \left( f_m( \{ p_T^{\text{gen}}, \eta^{\text{gen}}, \phi^{\text{gen}}, m^{\text{gen}} , \tau \}, m^\text{reco} ) $$
# 
# # Train Mass
# 
# for mass, 
# 
# $$\mathbf{y_m}=m_{\text{reco}}$$
# 
# and 
# 
# $$\mathbf{x_m}=\{p_T^{\text{gen}}, \eta^{\text{gen}}, \phi^{\text{gen}}, m^{\text{gen}} , \tau \}.$$
# 

# In[292]:


show_jupyter_image('images/IQN_training_flowchart.png',width=3000,height=1000)


# ### Batches, validation, losses, and plotting of losses functions

# In[293]:


def get_batch(x, t, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]
    batch_t = t[rows]
    # batch_x.T[-1] = np.random.uniform(0, 1, batch_size)
    return (batch_x, batch_t)

# Note: there are several average loss functions available 
# in pytorch, but it's useful to know how to create your own.
def average_quadratic_loss(f, t, x):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)

def average_cross_entropy_loss(f, t, x):
    # f and t must be of the same shape
    loss = torch.where(t > 0.5, torch.log(f), torch.log(1 - f))
    return -torch.mean(loss)

def average_quantile_loss(f, t, x):
    # f and t must be of the same shape
    tau = x.T[-1] # last column is tau.
    #Eq (2)
    return torch.mean(torch.where(t >= f, 
                                  tau * (t - f), 
                                  (1 - tau)*(f - t)))

# function to validate model during training.
def validate(model, avloss, inputs, targets):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        x = torch.from_numpy(inputs).float()
        t = torch.from_numpy(targets).float()
        # remember to reshape!
        o = model(x).reshape(t.shape)
    return avloss(o, t, x)


def plot_average_loss(traces, ftsize=18,save_loss_plots=False, show_loss_plots=True):
    
    xx, yy_t, yy_v, yy_v_avg = traces
    
    # create an empty figure
    fig = plt.figure(figsize=(6, 4.5))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.set_title("Average loss")
    
    ax.plot(xx, yy_t, 'b', lw=2, label='Training')
    ax.plot(xx, yy_v, 'r', lw=2, label='Validation')
    #ax.plot(xx, yy_v_avg, 'g', lw=2, label='Running average')

    ax.set_xlabel('Iterations', fontsize=ftsize)
    ax.set_ylabel('average loss', fontsize=ftsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    ax.legend(loc='upper right')
    if save_loss_plots:
        filename='IQNx4_%s_Loss.png' % target 
        mkdir('images/loss_plots')
        PATH = os.path.join(IQN_BASE, 'images', 'loss_plots', filename)
        plt.savefig(PATH)
        print('\nloss curve saved in %s' % PATH)
    if show_loss_plots:
        show_plot()


# # Get training and testing features and targets

# In[294]:


def split_t_x(df, target, input_features):
    """ Get teh target as the ratio, according to the T equation"""
    
    if target=='RecoDatam':
        t = T('m', scaled_df=train_data_m)
    if target=='RecoDatapT':
        t = T('pT', scaled_df=train_data_m)
    if target=='RecoDataeta':
        t = T('eta', scaled_df=train_data_m)
    if target=='RecoDataphi':
        t = T('phi', scaled_df=train_data_m)
    x = np.array(df[input_features])
    return np.array(t), x


# In[295]:


print('Features = ', features)
print('\nTarget = ', target)


# In[296]:


print(f'spliting data for {target}')
train_t_ratio, train_x = split_t_x(df= train_data_m, target = target, input_features=features)
print('train_t shape = ',train_t_ratio.shape , 'train_x shape = ', train_x.shape)
print('\n Training features:\n')
print(train_x)
valid_t_ratio, valid_x = split_t_x(df= test_data_m, target = target, input_features=features)
print('valid_t shape = ',valid_t_ratio.shape , 'valid_x shape = ', valid_x.shape)

print('no need to train_test_split since we already have the split dataframes')


# In[297]:


print(valid_x.mean(axis=0), valid_x.std(axis=0))
print(train_x.mean(axis=0), train_x.std(axis=0))


# we expect the targets to have mean 0 and variance=1, since theyre the only things standarized

# In[298]:


print(valid_t_ratio.mean(), valid_t_ratio.std())
print(train_t_ratio.mean(), train_t_ratio.std())


# ### Aplly final $z$ to the train and test set features, but run it only once! (generator)

# In[299]:


# def z__scale_targets(train_t_ratio, valid_t_ratio):
#     print('##########################################\n')
#     print('BEFORE SCALING')
    
#     #yield train_t_ratio, valid_t_ratio
    


# In[300]:


NFEATURES=train_x.shape[1]
for i in range(NFEATURES-1):
    train_x[:,i] = z(train_x[:,i])
    valid_x[:,i] = z(valid_x[:,i])
    
print(valid_x.mean(axis=0), valid_x.std(axis=0))

print(train_x.mean(axis=0), train_x.std(axis=0))


# In[301]:


train_x


# ### Apply $z$ to targets before training

# In[302]:


train_t_ratio = z(train_t_ratio) 
valid_t_ratio= z(valid_t_ratio)

print(valid_t_ratio.mean(), valid_t_ratio.std())
print(train_t_ratio.mean(), train_t_ratio.std())


# In[303]:


fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(autoscale_on=True)
ax.grid()
for i in range(NFEATURES):
    ax.hist(train_x[:,i], alpha=0.35, label=f'feature {i}' )
    set_axes(ax=ax, xlabel="Transformed features X' ",title="training features post-z score: X'=z(L(X))")


# In[304]:


train_x[:,-1].max()


# ### Training and running-of-training functions

# ### Define basic NN model

# In[336]:


class RegularizedRegressionModel(nn.Module):
    #inherit from the super class
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size, dropout):
        super().__init__()
        layers = []
        for _ in range(nlayers):
            if len(layers) ==0:
                #inital layer has to have size of input features as its input layer
                #its output layer can have any size but it must match the size of the input layer of the next linear layer
                #here we choose its output layer as the hidden size (fully connected)
                layers.append(nn.Linear(nfeatures, hidden_size))
                #batch normalization
                layers.append(nn.BatchNorm1d(hidden_size))
                #dropout only in the first layer
                #Dropout seems to worsen model performance
                layers.append(nn.Dropout(dropout))
                #ReLU activation 
                layers.append(nn.LeakyReLU())
            else:
                #if this is not the first layer (we dont have layers)
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                #Dropout seems to worsen model performance
                # layers.append(nn.Dropout(dropout))
                layers.append(nn.LeakyReLU())
                #output layer:
        layers.append(nn.Linear(hidden_size, ntargets)) 

        # only for classification add sigmoid
        # layers.append(nn.Sigmoid())
            #we have defined sequential model using the layers in oulist 
        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        return self.model(x)


# --------
# -------
# 
# ## Hyperparameter Training Workflow

# In[339]:


def get_tuning_sample():
    sample=int(200000)
    # train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample
    return train_x[:sample], train_t_ratio[:sample], valid_x[:sample], valid_t_ratio[:sample]
train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample = get_tuning_sample()
train_x_sample.shape


# In[432]:


class HyperTrainer():
    """loss, training and evaluation"""
    def __init__(self, model, optimizer, batch_size):
                 #, device):
        self.model = model
        #self.device= device
        self.optimizer = optimizer
        self.batch_size=batch_size
        self.n_iterations_tune=int(50)

        #the loss function returns the loss function. It is a static method so it doesn't need self
        # @staticmethod
        # def loss_fun(targets, outputs):
        #   tau = torch.rand(outputs.shape)
        #   return torch.mean(torch.where(targets >= outputs, 
        #                                   tau * (targets - outputs), 
        #                                   (1 - tau)*(outputs - targets)))

        #     This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
        #     by combining the operations into one layer

    def train(self, x, t):

        self.model.train()
        final_loss = 0
        for iteration in range(self.n_iterations_tune):
            self.optimizer.zero_grad()
            batch_x, batch_t = get_batch(x, t,  self.batch_size)#x and t are train_x and train_t

            # with torch.no_grad():
            inputs=torch.from_numpy(batch_x).float()
            targets=torch.from_numpy(batch_t).float()
            outputs = self.model(inputs)
            loss = average_quantile_loss(outputs, targets, inputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()

        return final_loss / self.batch_size

    def evaluate(self, x, t):

        self.model.eval()
        final_loss = 0
        for iteration in range(self.n_iterations_tune):
            batch_x, batch_t = get_batch(x, t,  self.batch_size)#x and t are train_x and train_t

            # with torch.no_grad():            
            inputs=torch.from_numpy(batch_x).float()
            targets=torch.from_numpy(batch_t).float()
            outputs = self.model(inputs)
            loss =average_quantile_loss(outputs, targets, inputs)
            final_loss += loss.item()
        return final_loss / self.batch_size

    
EPOCHS=1
def run_train(params, save_model=False):
    """For tuning the parameters"""

    model =  RegularizedRegressionModel(
              nfeatures=train_x_sample.shape[1], 
                ntargets=1,
                nlayers=params["nlayers"], 
                hidden_size=params["hidden_size"],
                dropout=params["dropout"]
                )
    # print(model)
    

    learning_rate= params["learning_rate"]
    optimizer_name = params["optimizer_name"]
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"]) 
    
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), 
                            lr=learning_rate, momentum = params["momentum"])
    
    trainer=HyperTrainer(model, optimizer, batch_size=params["batch_size"])
    best_loss = np.inf
    early_stopping_iter=10#stop after 10 iteractions of not improving loss
    early_stopping_coutner=0

    for epoch in range(EPOCHS):
        train_loss = trainer.train(train_x_sample, train_t_ratio_sample)
        valid_loss=trainer.evaluate(valid_x_sample, valid_t_ratio_sample)

        print(f"{epoch} \t {train_loss} \t {valid_loss}")
        if valid_loss<best_loss:
            best_loss=valid_loss
        else:
            early_stopping_coutner+=1
        if early_stopping_coutner > early_stopping_iter:
            break
            
    return best_loss

# run_train()

def objective(trial):
    params = {
          "nlayers": trial.suggest_int("nlayers",1,6),      
          "hidden_size": trial.suggest_int("hidden_size", 1, 16),
          "dropout": trial.suggest_float("dropout", 0.0,0.5),
          "optimizer_name" : trial.suggest_categorical("optimizer_name", ["RMSprop", "SGD"]),
          "momentum": trial.suggest_float("momentum", 0.0,0.9),
          "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-2),
          "batch_size": trial.suggest_int("batch_size", 500, 3000)

        }
    
    for step in range(10):

        temp_loss = run_train(params,save_model=False)
        trial.report(temp_loss, step)
        #activate pruning (early stopping)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return temp_loss

def tune_hyperparameters():
    print(f'Getting best hyperparameters for target {target}')
    # study=optuna.create_study(direction="minimize")
    #choose a different sampling strategy (https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html#optuna.samplers.CmaEsSampler)
    sampler=optuna.samplers.RandomSampler()
    # sampler=False
    if sampler:
        study=optuna.create_study(direction='minimize',
                                  pruner=optuna.pruners.MedianPruner(), sampler=sampler)
    else:
        study=optuna.create_study(direction='minimize',
                                  pruner=optuna.pruners.Hyperband())
    study.optimize(objective, n_trials=10)
    best_trial = study.best_trial
    print('best model parameters', best_trial.params)

    best_params=best_trial.params#this is a dictionary
    tuned_dir = os.path.join(IQN_BASE,'best_params')
    mkdir(tuned_dir)
    filename=os.path.join(tuned_dir,'best_params_Test_Trials.csv')
    param_df=pd.DataFrame({
                            'n_layers':best_params["nlayers"], 
                            'hidden_size':best_params["hidden_size"], 
                            'dropout':best_params["dropout"],
                            'optimizer_name':best_params["optimizer_name"],
                            'learning_rate': best_params["learning_rate"], 
                            'batch_size':best_params["batch_size"],
                            'momentum':best_params["momentum"]},
                                    index=[0]
    )

    param_df.to_csv(filename)   


# In[433]:


tune_hyperparameters()


# In[413]:


# BEST_PARAMS = {'nlayers': 6, 'hidden_size': 2, 'dropout': 0.40716885971031636, 'optimizer_name': 'Adam', 'learning_rate': 0.005215585403055171, 'batch_size': 1983}


# In[434]:


# def get_model_params_tuned()

tuned_dir = os.path.join(IQN_BASE,'best_params')

tuned_filename=os.path.join(tuned_dir,'best_params_Test_Trials.csv')
# BEST_PARAMS = pd.read_csv(os.path.join(IQN_BASE, 'best_params','best_params_Test_Trials.csv'))
BEST_PARAMS=pd.read_csv(tuned_filename)
print(BEST_PARAMS)

n_layers = int(BEST_PARAMS["n_layers"]) 
hidden_size = int(BEST_PARAMS["hidden_size"])
dropout = float(BEST_PARAMS["dropout"])

optimizer_name = BEST_PARAMS["optimizer_name"]
print(type(optimizer_name))
optimizer_name = BEST_PARAMS["optimizer_name"].to_string().split()[1]


# In[435]:


NFEATURES=train_x.shape[1]

def load_untrained_model():
    model=RegularizedRegressionModel(nfeatures=NFEATURES, ntargets=1,
                               nlayers=n_layers, hidden_size=hidden_size, dropout=dropout)
    print(model)
    return model

model=load_untrained_model()


# In[436]:


# optimizer_name =  'Adam'
best_learning_rate =  float(BEST_PARAMS["learning_rate"])
momentum=float(BEST_PARAMS["momentum"])
best_optimizer_temp = getattr(torch.optim, optimizer_name)(model.parameters(), lr=best_learning_rate,momentum=momentum)
batch_size = int(BEST_PARAMS["batch_size"])


# In[437]:


best_optimizer_temp


# In[439]:


@debug
def get_model_params_simple():
    dropout=0.2
    n_layers = 2
    n_hidden=32
    starting_learning_rate=1e-3
    print('n_iterations, n_layers, n_hidden, starting_learning_rate, dropout')
    return n_iterations, n_layers, n_hidden, starting_learning_rate, dropout

get_model_params_simple()


# ### Run training

# In[440]:


# BATCHSIZE=10000
BATCHSIZE=batch_size
def train(model, optimizer, avloss, getbatch,
          train_x, train_t, 
          valid_x, valid_t,
          batch_size, 
          n_iterations, traces, 
          step=10, window=10):
    
    # to keep track of average losses
    xx, yy_t, yy_v, yy_v_avg = traces
    
    n = len(valid_x)
    
    print('Iteration vs average loss')
    print("%10s\t%10s\t%10s" % \
          ('iteration', 'train-set', 'valid-set'))
    
    for ii in range(n_iterations):

        # set mode to training so that training specific 
        # operations such as dropout are enabled.
        model.train()
        
        # get a random sample (a batch) of data (as numpy arrays)
        batch_x, batch_t = getbatch(train_x, train_t, batch_size)
        # batch_x[:,-1]=batch_x[:,-1] 
        # convert the numpy arrays batch_x and batch_t to tensor 
        # types. The PyTorch tensor type is the magic that permits 
        # automatic differentiation with respect to parameters. 
        # However, since we do not need to take the derivatives
        # with respect to x and t, we disable this feature
        with torch.no_grad(): # no need to compute gradients 
            # wrt. x and t
            x = torch.from_numpy(batch_x).float()
            t = torch.from_numpy(batch_t).float()      

        # compute the output of the model for the batch of data x
        # Note: outputs is 
        #   of shape (-1, 1), but the tensor targets, t, is
        #   of shape (-1,)
        # In order for the tensor operations with outputs and t
        # to work correctly, it is necessary that they have the
        # same shape. We can do this with the reshape method.
        outputs = model(x).reshape(t.shape)
   
        # compute a noisy approximation to the average loss
        empirical_risk = avloss(outputs, t, x)
        
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer.zero_grad()       # clear previous gradients
        empirical_risk.backward()   # compute gradients
        
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer.step()            # move one step
        
        if ii % step == 0:
            
            acc_t = validate(model, avloss, train_x[:n], train_t[:n]) 
            acc_v = validate(model, avloss, valid_x[:n], valid_t[:n])
            yy_t.append(acc_t)
            yy_v.append(acc_v)
            
            # compute running average for validation data
            len_yy_v = len(yy_v)
            if   len_yy_v < window:
                yy_v_avg.append( yy_v[-1] )
            elif len_yy_v == window:
                yy_v_avg.append( sum(yy_v) / window )
            else:
                acc_v_avg  = yy_v_avg[-1] * window
                acc_v_avg += yy_v[-1] - yy_v[-window-1]
                yy_v_avg.append(acc_v_avg / window)
                        
            if len(xx) < 1:
                xx.append(0)
                print("%10d\t%10.6f\t%10.6f" % \
                      (xx[-1], yy_t[-1], yy_v[-1]))
            else:
                xx.append(xx[-1] + step)
                    
                print("\r%10d\t%10.6f\t%10.6f\t%10.6f" % \
                          (xx[-1], yy_t[-1], yy_v[-1], yy_v_avg[-1]), 
                      end='')
            
    print()      
    return (xx, yy_t, yy_v, yy_v_avg)


# @time_type_of_func(tuning_or_training='training')
def run(model, 
        train_x, train_t, 
        valid_x, valid_t, traces,
        n_batch=BATCHSIZE, 
        n_iterations=n_iterations, 
        traces_step=200, 
        traces_window=200,
        save_model=False):

    learning_rate= best_learning_rate
    #add weight decay
    L2=1e-3
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=best_learning_rate, momentum=momentum)
    
    #starting at 10^-3	    
    traces = train(model, optimizer, 
                      average_quantile_loss,
                      get_batch,
                      train_x, train_t, 
                      valid_x, valid_t,
                      n_batch, 
                  n_iterations,
                  traces,
                  step=traces_step, 
                  window=traces_window)
    
    learning_rate=learning_rate/100
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    # #10^-4
    traces = train(model, optimizer, 
                      average_quantile_loss,
                      get_batch,
                      train_x, train_t, 
                      valid_x, valid_t,
                      n_batch, 
                  n_iterations,
                  traces,
                  step=traces_step, 
                  window=traces_window)


    learning_rate=learning_rate/100
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    #10^-6
    traces = train(model, optimizer, 
                      average_quantile_loss,
                      get_batch,
                      train_x, train_t, 
                      valid_x, valid_t,
                      n_batch, 
                  n_iterations,
                  traces,
                  step=traces_step, 
                  window=traces_window)

    # plot_average_loss(traces)

    if save_model:
        filename='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
        PATH = os.path.join(IQN_BASE, 'trained_models', filename)
        torch.save(model.state_dict(), PATH)
        print('\ntrained model dictionary saved in %s' % PATH)
    return  model


# ## See if trainig works on T ratio

# In[442]:


IQN_trace=([], [], [], [])
traces_step = 800
traces_window=traces_step
n_iterations=10000
IQN = run(model=model,train_x=train_x, train_t=train_t_ratio, 
        valid_x=valid_x, valid_t=valid_t_ratio, traces=IQN_trace, n_batch=BATCHSIZE, 
        n_iterations=n_iterations, traces_step=traces_step, traces_window=traces_window,
        save_model=False)



# ## Save trained model (if its good, and if you haven't saved above) and load trained model (if you saved it)

# In[443]:


filename='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
trained_models_dir='trained_models'
mkdir(trained_models_dir)
PATH = os.path.join(IQN_BASE,trained_models_dir , filename)

@debug
def save_model(model):
    print(model)
    torch.save(model.state_dict(), PATH)
    print('\ntrained model dictionary saved in %s' % PATH)

@debug
def load_model(PATH):
    # n_layers = int(BEST_PARAMS["n_layers"]) 
    # hidden_size = int(BEST_PARAMS["hidden_size"])
    # dropout = float(BEST_PARAMS["dropout"])
    # optimizer_name = BEST_PARAMS["optimizer_name"].to_string().split()[1]
    # learning_rate =  float(BEST_PARAMS["learning_rate"])
    # batch_size = int(BEST_PARAMS["batch_size"])
    model =  RegularizedRegressionModel(
        nfeatures=train_x.shape[1], 
        ntargets=1,
        nlayers=n_layers, 
        hidden_size=hidden_size, 
        dropout=dropout
        )
    model.load_state_dict(torch.load(PATH) )
    #OR
    #model=torch.load(PATH)#BUT HERE IT WILL BE A DICT (CANT BE EVALUATED RIGHT AWAY) DISCOURAGED!
    model.eval()
    print(model)
    return model


# In[444]:


save_model(IQN)


# In[445]:


IQN


# In[446]:


plt.hist(valid_t_ratio, label='post-z ratio target');
for i in range(NFEATURES):
    plt.hist(valid_x[:,i], label =f"feature {i}", alpha=0.35)
plt.legend();plt.show()


# In[447]:


def simple_eval(model):
    model.eval()
    valid_x_tensor=torch.from_numpy(valid_x).float()
    pred = IQN(valid_x_tensor)
    p = pred.detach().numpy()
    fig, ax = plt.subplots(1,1)
    label=FIELDS[target]['ylabel']
    ax.hist(p, label=f'Predicted post-z ratio for {label}', alpha=0.4, density=True)
    orig_ratio = z(T('m', scaled_df=train_data_m))
    print(orig_ratio[:5])
    ax.hist(orig_ratio, label = f'original post-z ratio for {label}', alpha=0.4,density=True)
    
    set_axes(ax, xlabel='predicted $T$')
    print('predicted ratio shape: ', p.shape)
    return p
    
p = simple_eval(IQN)
plt.show()


# In[397]:


# IQN.eval()
# valid_x_tensor=torch.from_numpy(valid_x).float()
# pred = IQN(valid_x_tensor)
# p = pred.detach().numpy()
# plt.hist(p, label='predicted $T$ ratio');plt.legend();plt.show()


# $$
#         f_{\text{IQN}} (\mathcal{O}) =  z \left( \frac{\mathbb{L} (\mathcal{O}^{\text{reco}}) +10 }{\mathbb{L}(\mathcal{O}^{\text{gen}}) +10} \right),
# $$
# 
# 
# So, to de-scale, (for our observable $\mathcal{O}=m$ ),
# 
# $$
#     m^{\text{predicted}} = \mathbb{L}^{-1} \left[ z^{-1} (f_{\text{IQN}} ) \left[ \mathbb{L} (m^\text{gen})+10 \right] -10 \right]
# $$
# 
# Note that $z^{-1} (f_{\text{IQN}} )$ should use the mean and std of the ratio thing for the target 
# 
# $$z^{-1} (f_{\text{IQN}} ) = z^{-1}\left( y_{pred}, \text{mean}=\text{mean}(\mathbb{T}(\text{target_variable})), std=std (\mathbb{T}(\text{target_variable} ) \right)$$

# In[398]:


def z_inverse(xprime, mean, std):
    return xprime * std + mean


# In[399]:


recom_unsc_mean=TEST_SCALE_DICT[target]['mean']
recom_unsc_std=TEST_SCALE_DICT[target]['std']
print(recom_unsc_mean,recom_unsc_std)


# Get unscaled dataframe again, just to verify

# In[400]:


raw_train_data=pd.read_csv(os.path.join(DATA_DIR,'train_data_10M_2.csv'),
                      usecols=all_cols,
                      nrows=SUBSAMPLE
                      )

raw_test_data=pd.read_csv(os.path.join(DATA_DIR,'test_data_10M_2.csv'),
                      usecols=all_cols,
                     nrows=SUBSAMPLE
                     )
raw_test_data.describe()


# In[239]:


m_reco = raw_test_data['RecoDatam']
m_gen = raw_test_data['genDatam']
plt.hist(m_reco,label=r'$m_{gen}^{test \ data}$');plt.legend();plt.show()


# 
# 
# Apply the descaling formula for our observable
# 
# $$
#     m^{\text{predicted}} = \mathbb{L}^{-1} \left[ z^{-1} (f_{\text{IQN}} ) \left[ \mathbb{L} (m^\text{gen})+10 \right] -10 \right]
# $$
# 
# * First, calculate $z^{-1} (f_{\text{IQN}} )$

# In[401]:


print(valid_t_ratio.shape, valid_t_ratio[:5])


# In[402]:


orig_ratio = T('m', scaled_df=train_data_m)
orig_ratio[:5]


# In[403]:


z_inv_f =z_inverse(xprime=p, mean=np.mean(orig_ratio), std=np.std(orig_ratio))
z_inv_f[:5]


# * Now 
# 
# $$\mathbb{L}(\mathcal{O^{\text{gen}}}) = \mathbb{L} (m^{\text{gen}})$$
# 

# In[404]:


L_obs = L(orig_observable=m_gen, label='m')
L_obs[:5]


# In[405]:


print(L_obs.shape, z_inv_f.shape)


# In[406]:


z_inv_f = z_inv_f.flatten();print(z_inv_f.shape)


# * "factor" $ = z^{-1} (f_{\text{IQN}} ) \left[ \mathbb{L} (m^\text{gen})+10 \right] -10 $

# In[407]:


factor = (z_inv_f * (L_obs  + 10) )-10
factor[:5]


# In[408]:


m_pred = L_inverse(L_observable=factor, label='m')
# pT_pred=get_finite(pT_pred)


# In[409]:


m_pred


# In[410]:


plt.hist(m_pred.flatten(),label='predicted',alpha=0.3);
plt.hist(m_reco,label=r'$m_{reco}^{test \ data}$',alpha=0.3);

plt.legend();plt.show()


# ------------------
# ### Paper plotting

# In[369]:


range_=[0,25]
bins=50
data=raw_train_data
YLIM=(0.8,1.2)
data = data[['RecoDatapT','RecoDataeta','RecoDataphi','RecoDatam']]
data.columns = ['realpT','realeta','realphi','realm']
REAL_DIST=data['realm']
norm_data=data.shape[0]
AUTOREGRESSIVE_DIST = m_pred
norm_IQN=AUTOREGRESSIVE_DIST.shape[0]
norm_autoregressive=AUTOREGRESSIVE_DIST.shape[0]
norm_IQN=norm_autoregressive
print('norm_data',norm_data,'\nnorm IQN',norm_IQN,'\nnorm_autoregressive', norm_autoregressive)


# In[370]:


def get_hist(label):
    """label could be "pT", "eta", "phi", "m"
    """
    predicted_label_counts, label_edges = np.histogram(JETS_DICT['Predicted_RecoData'+label]['dist'], 
    range=JETS_DICT['Predicted_RecoData'+label]['range'], bins=bins)
    real_label_counts, _ = np.histogram(JETS_DICT['Real_RecoData'+label]['dist'], 
    range=JETS_DICT['Real_RecoData'+label]['range'], bins=bins)
    label_edges = label_edges[1:]/2+label_edges[:-1]/2

    return real_label_counts, predicted_label_counts, label_edges

def get_hist_simple(label):
    predicted_label_counts, label_edges = np.histogram(m_pred , range=range_, bins=bins)
    real_label_counts, _ = np.histogram(REAL_DIST, range=range_, bins=bins)
    label_edges = label_edges[1:]/2+label_edges[:-1]/2
    return real_label_counts, predicted_label_counts, label_edges


# In[371]:


real_label_counts_m, predicted_label_counts_m, label_edges_m = get_hist_simple('m')


# In[372]:


def plot_one_m():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
    ax1.step(label_edges_m, real_label_counts_m/norm_data, where="mid", color="k", linewidth=0.5)# step real_count_pt
    ax1.step(label_edges_m, predicted_label_counts_m/norm_IQN, where="mid", color="#D7301F", linewidth=0.5)# step predicted_count_pt
    ax1.scatter(label_edges_m, real_label_counts_m/norm_data, label="reco",  color="k",facecolors='none', marker="o", s=5, linewidth=0.5)
    ax1.scatter(label_edges_m,predicted_label_counts_m/norm_IQN, label="predicted sbatch 1", color="#D7301F", marker="x", s=5, linewidth=0.5)
    ax1.set_xlim(range_)
    ax1.set_ylim(0, max(predicted_label_counts_m/norm_IQN)*1.1)
    ax1.set_ylabel("counts")
    ax1.set_xticklabels([])
    ax1.legend(loc='upper right')

    ratio=(predicted_label_counts_m/norm_IQN)/(real_label_counts_m/norm_data)
    ax2.scatter(label_edges_m, ratio, color="r", marker="x", s=5, linewidth=0.5)#PREDICTED (IQN)/Reco (Data)
    ax2.scatter(label_edges_m, ratio/ratio, color="k", marker="o",facecolors="none", s=5, linewidth=0.5)
    ax2.set_xlim(range_)
    # ax2.set_xlabel(labels[3])
    ax2.set_ylabel(r"$\frac{\textnormal{predicted}}{\textnormal{reco}}$")
    ax2.set_ylim((YLIM))
    ax2.set_xlim(range_)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.5, hspace=0.2)
    fig.subplots_adjust(wspace=0.0, hspace=0.1)
    # plt.savefig(DIR+'AUTOREGRESSIVE_m_TUNEND_MLP_OCT_18.pdf')
    #   plt.savefig('images/all_m_g2r.pdf')
    plt.show(); 
    # fig.show()

    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])


# In[373]:


plot_one_m()


# -------------
# -------------
# -------------
# 

# In[137]:


if target== 'RecoDatapT':
    label= '$p_T$ [GeV]'
    x_min, x_max = 20, 60
elif target== 'RecoDataeta':
    label = '$\eta$'
    x_min, x_max = -5.4, 5.4
elif target =='RecoDataphi':
    label='$\phi$'
    x_min, x_max = -3.4, 3.4
elif target == 'RecoDatam':
    label = ' $m$ [GeV]'
    x_min, x_max = 0, 18


    
def evaluate_model(dnn, target, src,
               fgsize=(6, 6), 
               ftsize=20,save_image=False, save_pred=False,
               show_plot=True):
    eval_data=pd.read_csv(os.path.join(DATA_DIR,'test_data_10M_2.csv'))
    ev_features=X
    #['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','tau']
    
    eval_data=eval_data[ev_features]
    
    print('EVALUATION DATA OLD INDEX\n', eval_data.head())

    

                            
    dnn.eval()
    y = dnn(eval_data)
    eval_data['RecoDatam']=y
    new_cols= ['RecoDatam'] + X
    eval_data=eval_data.reindex(columns=new_cols)
    print('EVALUATION DATA NEW INDEX\n', eval_data.head())

    eval_data.to_csv('AUTOREGRESSIVE_m_Prime.csv')


    if save_pred:
        pred_df = pd.DataFrame({T+'_predicted':y})
        pred_df.to_csv('predicted_data/dataset2/'+T+'_predicted_MLP_iter_5000000.csv')
        
    if save_image or show_plot:
        gfile ='fig_model_%s.png' % target
        xbins = 100
        xmin  = src['xmin']
        xmax  = src['xmax']
        xlabel= src['xlabel']
        xstep = (xmax - xmin)/xbins

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fgsize)
        
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(xlabel, fontsize=ftsize)
        ax.set_xlabel('reco jet '+label, fontsize=ftsize)
        ax.set_ylabel(y_label_dict[target], fontsize=ftsize)

        ax.hist(train_data['RecoDatam'], 
                bins=xbins, 
                range=(xmin, xmax), 
                alpha=0.3, 
                color='blue', 
                density=True, 
                label='simulation')
        ax.hist(y, 
                bins=xbins, 
                range=(xmin, xmax), 
                alpha=0.3, 
                color='red', 
                density=True, 
                label='$y^\prime$')
        ax.grid()
        ax.legend()
        
        
        if save_image:
            plt.savefig('images/'+T+'IQN_Consecutive_'+N+'.png')
            print('images/'+T+'IQN_Consecutive_'+N+'.png')
        if show_plot:
            plt.tight_layout()
            plt.show()
##########
################################################CNN







def main():
    start=time.time()
    print('estimating mass\n')
    model =  utils.RegularizedRegressionModel(nfeatures=train_x.shape[1], ntargets=1,nlayers=n_layers, hidden_size=n_hidden)
    traces = ([], [], [], [])
    dnn = run(model, scalers, target, train_x, train_t, valid_x, valid_t, traces)
    evaluate_model( dnn, target, source)



if __name__ == "__main__":
    main()



# # Plot predicted vs real reco (in our paper's format)

# In[ ]:





# # Train $p_T$ using saved variables above

# In[ ]:





# Evaluate $p_T$ and save predicted distribution

# In[ ]:





# Plot reco $p_T$ and  predicted reco $p_T$ marginal densities

# In[155]:


# show_jupyter_image('screenshot.png')


# <!-- > I guess it works now -->

# commented new ideas below

# <!-- ### Ideas for a future paper
# 
# me and Harrison would like to use this method for on-the-fly stochastic folding of events in MC generators (potentially even including CMSSW formats like [nanoaod](https://github.com/cms-nanoAOD/nanoAOD-tools), such as in Madminer (but using IQN as opposed to Delphes for detector simulation) for any observable. This also beings the possibility of using LFI methods for much better inference on models (such as SMEFT) using any observable post-detector simulation. If you're interested in helping out on this, me and Harrison would like to do most of the code/ideas, but your occasional ideas/input would be incredibly valuable! -->

# In[181]:





# In[ ]:




