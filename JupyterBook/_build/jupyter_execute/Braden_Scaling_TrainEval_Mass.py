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

# In[2]:


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


# In[3]:


# os.environ['IQN_BASE']='/home/ali/Desktop/Pulled_Github_Repositories/torchQN'
# os.environ['DATA_DIR']='/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/data'


# ### A user is competent enought to do `source setup.sh` on a `setup.sh` script that comes in the repo, such as the next cell uncommented

# In[4]:


# %%writefile setup.sh
# #!/bin/bash
# export IQN_BASE=/home/ali/Desktop/Pulled_Github_Repositories/torchQN
# #DAVIDSON
# #export DATA_DIR='/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/data'
# #LOCAL
# export DATA_DIR='/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data'
# echo 'DATA DIR'
# ls -l $DATA_DIR
# #ln -s $DATA_DIR $IQN_BASE, if you want
# #conda create env -n torch_env -f torch_env.yml
# #conda activate torch_env
# mkdir -p ${IQN_BASE}/images/loss_plots ${IQN_BASE}/trained_models  ${IQN_BASE}/hyperparameters ${IQN_BASE}/predicted_data
# tree $IQN_BASE


# In[5]:


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

# In[6]:


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

if JUPYTER:
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

dropout=0.2

def get_model_params():
    return n_iterations, n_layers, n_hidden, starting_learning_rate, dropout


# ### Import the numpy data, convert to dataframe and save (if you haven't saved the dataframes)

# # Explore the Dataframe and preprocess

# another flowchart for how IQN works autoregressively to get $p_T'$, etc

# Plotting

# In[7]:


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
    plt.show()


# In[8]:


use_svg_display()
show_jupyter_image('images/pythia_ppt_diagram.png', width=2000,height=500)


# <!-- For Davidson team, please read try to all the code/comments before asking me questions! -->

# In[9]:


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

###############################################################################################
y_label_dict ={'RecoDatapT':'$p(p_T)$'+' [ GeV'+'$^{-1} $'+']',
                    'RecoDataeta':'$p(\eta)$', 'RecoDataphi':'$p(\phi)$',
                    'RecoDatam':'$p(m)$'+' [ GeV'+'$^{-1} $'+']'}

loss_y_label_dict ={'RecoDatapT':'$p_T^{reco}$',
                    'RecoDataeta':'$\eta^{reco}$', 'RecoDataphi':'$\phi^{reco}$',
                    'RecoDatam':'$m^{reco}$'}


# In[11]:


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


# In[12]:


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


# In[13]:


explore_data(df=train_data, title='Unscaled Dataframe')


# In[131]:


print(train_data.shape)
train_data.describe()#unscaled


# In[132]:


print(test_data.shape)
test_data.describe()#unscaled


# In[19]:


# np.array(train_data['genDatapT'])


# standarization is someimes done in the following way:
# $$ X' = \frac{X-X_{min}}{X_{max}-X_{min}} \qquad \rightarrow \qquad X= X' (X_{max}-X_{min}) + X_{min}$$

# In[16]:


# def standarize(values):
#     expected_min, expected_max = values.min(), values.max()
#     scale_factor = expected_max - expected_min
#     offset = expected_min
#     standarized_values = (values - offset)/scale_factor 
#     return standarized_values


# Or by taking z-score:
# $$ X'=\frac{X-E[X]}{\sigma_{X}}  \qquad \rightarrow \qquad X = X' \sigma_{X} + E[X]$$

# ## Results prior to Braden-scaling
# 
# Recall that the best IQNx4 autoregressive results that I attained prior to trying the Braden scaling was the following (which was implemented in the Davidson cluster here: `/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/OCT_7/*.py` and copied to my repo [here](https://github.com/AliAlkadhim/torchQN/tree/master/OCT_7)):
# 

# In[23]:


show_jupyter_image('OCT_7/AUTOREGRESSIVE_RESULTS_OCT7.png',width = 800, height = 100)


# # Braden Scaling
# 
# ## Scale the data accoding to the "Braden Kronheim scaling" :
# 
# $$\mathbb{T}(p_T) = z(\log p_T), \qquad \mathbb{T}(\eta) = z(\eta), \qquad \mathbb{T}(\phi) = z(\phi), \qquad \mathbb{T}(m) = z(\log (m + 2))$$ 
# 
# $$ \mathbb{T}(\tau) = 6\tau - 3 $$

# For a jet observable $\mathcal{O}$ (or $\tau$), the data is first scaled according to:
# 
# $$
# \begin{align}
#     \mathbb{L} (\mathcal{O}) &=
#     \begin{cases}
#         \log{\mathcal{O}}, \qquad & \text{if } \mathcal{O}= p_T \\
#         \mathcal{O}, \qquad & \text{if } \mathcal{O}=\eta \\
#         \log (\mathcal{O} + 2), \qquad & \text{if } \mathcal{O}=m \\
#         \mathcal{O}, \qquad & \text{if } \mathcal{O}=\phi \\
#         6 \mathcal{O} -3, \qquad & \text{if } \mathcal{O}=\tau
#     \end{cases}
# \end{align}
# $$
# 
# Then the predicted target for a desired reco observable $\mathcal{O}$ is chosen to be the following function
# 
# $$
#     \mathbb{T}(O) = z \left( \frac{\mathbb{L} (\mathcal{O}^{\text{reco}}) +10 }{\mathbb{L}(\mathcal{O}^{\text{gen}}) +10} \right),
# $$
# 
# where for a random variable $x$, $z$ is the standardization function (z-score):
# 
# $$
#    x'= z (x) \equiv \frac{x-\bar{x}}{\sigma_{x}}.
# $$
# 
# Such that its inverse is 
# $$ z^{-1}(x') = x' \ \sigma_x + \bar{x} $$
# If we do this on the data, after training, the NN $f_{\text{IQN}}$ will not estimate the observable, $\mathcal{O}^{\text{predicted}} \ne \mathcal{O}^{\text{reco}}$, but will instead estimate 
# 
# $$
#         f_{\text{IQN}} (\mathcal{O}) =  z \left( \frac{\mathbb{L} (\mathcal{O}^{\text{reco}}) +10 }{\mathbb{L}(\mathcal{O}^{\text{gen}}) +10} \right),
# $$
# 
# which needs to be de-scaled (when evaluated on the data that which has been scaled according to $\mathbb{T}(\text{evaluation data}) = z \left( \frac{\mathbb{L} (\text{data}^{\text{reco}}) +10 }{\mathbb{L}(\text{data}^{\text{gen}}) +10} \right) $
# )
# in order to copare with $\mathcal{O}$ directly. The descaling for $\mathcal{O}=p_T$ (as an example) would be:
# $$
#     p_T^{\text{predicted}} = \mathbb{L}^{-1} \left[ z^{-1} (f_{\text{IQN}} ) \left[ \mathbb{L} (p_T^\text{gen})+10 \right] -10 \right]
# $$

# -----------
# 
# ## Basically the scaling procedure is the following:
# 
# 1. Split the data into train and test dataframes
# 2. fit the scaler on each of the train and test sets independently, that is, get the mean and std, ( optionally and min and max or other quantities) of each feature (column) of each of the train and test dataframes, independently.
# 3. transform each of the train and test sets independently. That is, use the means and stds of each column to transform a column $X$ into a column $X'$ e.g. according to 
# $$ X'=z(X)= \frac{X-E[X]}{\sigma_{X}}$$
# 4. Train NN on transformed features $X_{train}'$ (and target $y_{train}'$) (in train df, but validate on test set, which will not influence the weights of NN ( just used for observation that it doesnt overfit) )
# 5. Once the training is done, *evaluate the NN on transformed features of the test set* $X_{test}'$, i.e. do $NN(X_{test}')$, which will result in a scaled prediction of the target $y_{pred}'$
# 6. Unscale the $y_{pred}'$, i.e. apply the inverse of the scaling operation, e.g.
# $$ y_{pred}=z^{-1}(y_{pred}')= y_{pred}' \sigma{y} + E[y]$$,
# where $\sigma{y}$ and $E[y]$ are attained from the test set *prior to training and scaling*.
# 7. Compare to $y$ (the actual distribution you're trying to estimate) one-to-one

# In[15]:


use_svg_display()
show_jupyter_image('images/scaling_forNN.jpg', width=2000,height=500)


# In[91]:


def z(x):
    eps=1e-20
    return (x - np.mean(x))/(np.std(x)+ eps)
def z_inverse(xprime, x):
    return xprime * np.std(x) + np.mean(x)


# In[40]:


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


# In[48]:


TRAIN_SCALE_DICT = get_scaling_info(train_data);print(TRAIN_SCALE_DICT)
print('\n\n')
TEST_SCALE_DICT = get_scaling_info(test_data);print(TEST_SCALE_DICT)


# In[98]:


def L(orig_observable, label):
    eps=1e-10
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
        L_observable = (6*orig_observable) - 3
    
    return L_observable.to_numpy()


# In[99]:


def L_inverse(L_observable, label):
    eps=1e-10
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
        L_inverse_observable = (L_observable+3)/6
        
    if not isinstance(L_inverse_observable, np.ndarray):
        L_inverse_observable = L_inverse_observable.to_numpy()
    return L_inverse_observable


# In[100]:


def T(variable, scaled_df):
    if variable=='pT':
        L_pT_gen=scaled_df['genDatapT']
        L_pT_reco = scaled_df['RecoDatapT']
        target = z( (L_pT_reco+10)/(L_pT_gen+10) )
    if variable=='eta':
        L_eta_gen=scaled_df['genDataeta']
        L_eta_reco = scaled_df['RecoDataeta']
        target = z( (L_eta_reco+10)/(L_eta_gen+10) )
    if variable=='phi':
        L_phi_gen=scaled_df['genDataphi']
        L_phi_reco = scaled_df['RecoDataphi']
        target = z( (L_phi_reco+10)/(L_phi_gen+10) )
    if variable=='m':
        L_m_gen=scaled_df['genDatam']
        L_m_reco = scaled_df['RecoDatam']
        target = z( (L_m_reco+10)/(L_m_gen+10) )
    
    return target


# In[101]:


def L_scale_df(df, title, save=False):
    #scale
    SUBSAMPLE=int(1e4)
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


# In[103]:


scaled_train_data = L_scale_df(train_data, title='scaled_train_data_10M_2.csv',
                             save=False)
print('\n\n')
scaled_test_data = L_scale_df(test_data,  title='scaled_test_data_10M_2.csv',
                            save=False)

explore_data(df=scaled_train_data, title='Braden Kronheim-L-scaled Dataframe', scaled=True)


# In[104]:


labels = ['pT', 'eta','phi','m']
for label in labels:
    target = T(label, scaled_df=scaled_train_data)
    plt.hist(target, label = '$T($' +label+ '$)$' )
# target_pT = T('pT', scaled_df=scaled_train_data)
# target_eta = T('eta', scaled_df=scaled_train_data)
# target_phi = T('phi', scaled_df=scaled_train_data)
# target_m = T('m', scaled_df=scaled_train_data)
# plt.hist(target_pT, label='$T(p_T)$')

plt.legend();plt.show()


# In[105]:


# scaled_train_data = L_scale_df(train_data, title='scaled_train_data_10M_2.csv',
#                              save=True)
# print('\n\n')
# scaled_test_data = L_scale_df(test_data,  title='scaled_test_data_10M_2.csv',
#                             save=True)

# explore_data(df=scaled_train_data, title='Braden Kronheim-$L$-scaled Dataframe: standarize_IQN', scaled=True)


# ---------
# ------
# # ML
# 
# Note that this ideas is very powerful and has the potential to replace the use of Delphes/GEANT for most people. According to the [previous paper](https://arxiv.org/pdf/2111.11415.pdf) this method already works for a single IQN.
# 
# It's important to remember "the master formula" of all of machine learning:
# 
# $$\int \frac{\partial L}{\partial f} p(y|x) dy  = 0 \tag{1}$$
# 
# or, equivalently, 
# 
# $$ \frac{\delta R}{\delta f}=0,$$
# 
# where $L$ is the loss function, $f$ is the model (in this case IQN) (implicitly parameterized by potentially a  gazillion parameters), $y$ is the target(s) that we want to estimate, $x$ is the (set of) training features, $R$ is the risk functional:
# 
# $$ R[f] = \int \cdots \int \, p(t, \mathbf{x}) \, L(t, f(\mathbf{x}, \theta)) \, dt \, d\mathbf{x}$$
# 
# 
# So, for IQNs,
# 
# $$ L_{\text{IQN}}(f, y)=\left\{\begin{array}{ll}
# \tau(y-f(\boldsymbol{x}, \tau ; \boldsymbol{\theta})) & y \geq f(\boldsymbol{x}, \tau ; \boldsymbol{\theta}) \\
# (1-\tau)(f(\boldsymbol{x}, \tau ; \boldsymbol{\theta})-y) & y<f(\boldsymbol{x}, \tau ; \boldsymbol{\theta})
# \end{array},\right.$$
# 
# Means that what was done previously is that the risk functional, which is generally a functional of many models $f$, was a only a functional of a single model: $R[f_1,..., f_n] = f[f_1]$. Here we have 4 models 
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
# 
# $$
# 
# 
# 
# $$ \begin{align}
# R_{\text{IQN}x4} &= \int L_\text{IQN} \left( f_m (\mathbf{x_m},\tau), \mathbf{y_m} \right) p(\mathbf{x_m, y_m})  d \mathbf{x_m} d \mathbf{y_m} \\
# &\times \  ... \times \\ 
# &\times \int L_\text{IQN} \left( f_\phi (\mathbf{x_\phi},\tau), \mathbf{y_\phi} \right) p(\mathbf{x_\phi, y_\phi})  d \mathbf{x_\phi} d \mathbf{y_\phi}
# \end{align},$$
# 
# where, again, each model $f_i$ is also dependent on a set of parameters $\theta_i$ (dropped for simplicity)

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
# $$\int d \mathbf{x_m} d \mathbf{y_m} d \mathbf{\tau} \ p(\mathbf{x_m, y_m,\tau})   \ \frac{ \delta L_\text{IQN} \left( f_m (\mathbf{x_m},\tau), \mathbf{y_m} \right) }{\delta f_m} = 0$$
# 
# and by Eq (2)
# 
# $$
# \int d \mathbf{x_m} d \mathbf{y_m} d \mathbf{\tau} \ p(\mathbf{x_m, y_m,\tau})   \ \frac{ \delta L_\text{IQN} \left( f_m (\mathbf{x_m},\tau), \mathbf{y_m} \right) }{\delta f_m} = 0 \tag{7}
# $$
# >> ...
# <br>
# 
# Expand Eq (2) in Eq (7) and integrate wrt y to see that  $f(\mathbf{x},\mathbf{\tau})$ is the quantile function for $p(\mathbf{y}|\mathbf{x})$, i.e. (I believe) that IQNx4 should work basically exactly.

# $$R_{\text{IQN}x4} = [ L \left( f_m( \{ p_T^{\text{gen}}, \eta^{\text{gen}}, \phi^{\text{gen}}, m^{\text{gen}} , \tau \}, m^\text{reco} ) $$
# # Train Mass
# 
# for mass, $\mathbf{y_m}=m_{\text{reco}}$ and $\mathbf{x_m}=\{p_T^{\text{gen}}, \eta^{\text{gen}}, \phi^{\text{gen}}, m^{\text{gen}} , \tau \}$.
# 

# In[106]:


show_jupyter_image('images/IQN_training_flowchart.png',width=2000,height=600)


# ### Batches, validation, losses, and plotting of losses functions

# In[107]:


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


def mkdir(dir_):
    """make a directory without overwriting what's in it if it exists"""
    # assert isinstance(dir_, str)
    try:
        os.system('mkdir -p %s' % str(dir_) )
    except Exception:
        pass


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
        plt.savefig('images/loss_curves/IQN_'+N+T+'_Consecutive_2.png')
        print('\nloss curve saved in images/loss_curves/IQN_'+N+target+'_Consecutive.png')
    if show_loss_plots:
        plt.show()


# In[108]:


SUBSAMPLE=int(1e3)
target = 'RecoDatapT'
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


# ### Get training and testing features and targets

# In[127]:


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

def normal_split_t_x(df, target, input_features):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    # t = np.array(df[target])
    t = np.array(df[target])
    x = np.array(df[input_features])
    return t, x


# In[110]:


print(f'spliting data for {target}')
train_t_ratio, train_x = split_t_x(df= train_data_m, target = target, input_features=features)
print('train_t shape = ',train_t_ratio.shape , 'train_x shape = ', train_x.shape)
print('\n Training features:\n')
print(train_x)
valid_t_ratio, valid_x = split_t_x(df= test_data_m, target = target, input_features=features)
print('valid_t shape = ',valid_t_ratio.shape , 'valid_x shape = ', valid_x.shape)

print('no need to train_test_split since we already have the split dataframes')


# In[111]:


print(valid_x.mean(axis=0), valid_x.std(axis=0))
print(train_x.mean(axis=0), train_x.std(axis=0))


# we expect the targets to have mean 0 and variance=1, since theyre the only things standarized

# In[112]:


print(valid_t_ratio.mean(), valid_t_ratio.std())
print(train_t_ratio.mean(), train_t_ratio.std())


# ### Training and running-of-training functions

# In[113]:


n_iterations, n_layers, n_hidden, starting_learning_rate, dropout = get_model_params()
BATCHSIZE=1000
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


def run(model, target, 
        train_x, train_t, 
        valid_x, valid_t, traces,
        n_batch=BATCHSIZE, 
        n_iterations=n_iterations, 
        traces_step=10, 
        traces_window=10,
        save_model=False):

    learning_rate= starting_learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
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
    
    # learning_rate=learning_rate/10
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    # #10^-4
    # traces = train(model, optimizer, 
    #                   average_quantile_loss,
    #                   get_batch,
    #                   train_x, train_t, 
    #                   valid_x, valid_t,
    #                   n_batch, 
    #               n_iterations,
    #               traces,
    #               step=traces_step, 
    #               window=traces_window)


    # learning_rate=learning_rate/100
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    # #10^-6
    # traces = train(model, optimizer, 
    #                   average_quantile_loss,
    #                   get_batch,
    #                   train_x, train_t, 
    #                   valid_x, valid_t,
    #                   n_batch, 
    #               n_iterations,
    #               traces,
    #               step=traces_step, 
    #               window=traces_window)

    # plot_average_loss(traces)

    if save_model:
        filename='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
        PATH = os.path.join(IQN_BASE, 'trained_models', filename)
        torch.save(model.state_dict(), PATH)
        print('\ntrained model dictionary saved in %s' % PATH)
    #utils.ModelHandler(model, scalers)
    return  model


# ### Define basic NN model

# In[114]:


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
                # layers.append(nn.BatchNorm1d(hidden_size))
                #Dropout seems to worsen model performance
                # layers.append(nn.Dropout(dropout))
                #ReLU activation 
                layers.append(nn.ReLU())
            else:
                #if this is not the first layer (we dont have layers)
                layers.append(nn.Linear(hidden_size, hidden_size))
                # layers.append(nn.BatchNorm1d(hidden_size))
                #Dropout seems to worsen model performance
                # layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
                #output layer:
        layers.append(nn.Linear(hidden_size, ntargets)) 

        # only for classification add sigmoid
        # layers.append(nn.Sigmoid())
            #we have defined sequential model using the layers in oulist 
        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        return self.model(x)


# ### Run training

# In[115]:


def load_untrained_model():
    NFEATURES=train_x.shape[1]
    model=RegularizedRegressionModel(nfeatures=NFEATURES, ntargets=1,
                               nlayers=n_layers, hidden_size=n_hidden, dropout=dropout)
    return model
model=load_untrained_model()
print(model)


# ## See if trainig works on T ratio

# In[116]:


print(f'Training for {n_iterations} iterations')
start=time.time()
print('estimating %s\n' % target)
IQN_trace=([], [], [], [])
traces_step = 50
n_iterations=1000
IQN = run(model=model, target=target,train_x=train_x, train_t=train_t_ratio, 
        valid_x=valid_x, valid_t=valid_t_ratio, traces=IQN_trace, n_batch=1000, 
        n_iterations=n_iterations, traces_step=50, traces_window=50,
        save_model=False)

end=time.time()
difference=end-start
print('evaluating m took ',difference, 'seconds')


# In[123]:


IQN.eval()
valid_x_tensor=torch.from_numpy(valid_x).float()
pred = IQN(valid_x_tensor)
p = pred.detach().numpy()
plt.hist(p, label='using $T$ ratio');plt.legend();plt.show()


# In[122]:


def get_finite(values):
    return np.isfinite(values)

def z_inverse(xprime, unscaled_mean, unscaled_std):
    return xprime * unscaled_std + unscaled_mean


# Apparently not, but let's continue
# 
# $$
#     m^{\text{predicted}} = \mathbb{L}^{-1} \left[ z^{-1} (f_{\text{IQN}} ) \left[ \mathbb{L} (m^\text{gen})+10 \right] -10 \right]
# $$
# 

# In[118]:


recopt_unsc_mean=TEST_SCALE_DICT[target]['mean']
recop_unsc_std=TEST_SCALE_DICT[target]['std']
print(recopt_unsc_mean,recop_unsc_std)


# Get unscaled again, just to verify

# In[133]:


SUBSAMPLE=int(1e3)#subsample use for development - in production use whole dataset
train_data=pd.read_csv(os.path.join(DATA_DIR,'train_data_10M_2.csv'),
                      usecols=all_cols,
                      nrows=SUBSAMPLE
                      )

test_data=pd.read_csv(os.path.join(DATA_DIR,'test_data_10M_2.csv'),
                      usecols=all_cols,
                     nrows=SUBSAMPLE
                     )
test_data.describe()


# In[137]:


pT_reco = test_data['RecoDatapT']
pT_gen = test_data['genDatapT']
plt.hist(pT_reco);plt.show()


# In[141]:


p = pred.detach().numpy()
plt.hist(p);plt.show()


# In[142]:


z_inv_f =z_inverse(xprime=p, unscaled_mean=recopt_unsc_mean, unscaled_std=recop_unsc_std)

factor = z_inv_f * (L(orig_observable=pT_reco, label='pT') + 10) -10

pT_pred = L_inverse(L_observable=factor, label='pT')
# pT_pred=get_finite(pT_pred)


# In[146]:


pT_pred.flatten()


# In[145]:


plt.hist(get_finite(pT_pred.flatten()),bins=100);plt.show()


# ----

# In[152]:


print(f'spliting data for {target}')
train_t, train_x = normal_split_t_x(df= train_data_m, target = target, input_features=features)
print('train_t shape = ',train_t.shape , 'train_x shape = ', train_x.shape)
print('\n Training features:\n')
print(train_x)
valid_t, valid_x = normal_split_t_x(df= test_data_m, target = target, input_features=features)
print('valid_t shape = ',valid_t.shape , 'valid_x shape = ', valid_x.shape)

print('no need to train_test_split since we already have the split dataframes')


# In[148]:


print(valid_x.mean(axis=0), valid_x.std(axis=0))
print(train_x.mean(axis=0), train_x.std(axis=0))
print('\n\n')
print(valid_t.mean(), valid_t.std())
print(train_t.mean(), train_t.std())
plt.hist(train_t, bins=50);plt.hist(valid_t,bins=50);plt.show()


# In[149]:


n_iterations, n_layers, n_hidden, starting_learning_rate, dropout = get_model_params()
def load_untrained_model():
    NFEATURES=train_x.shape[1]
    model=RegularizedRegressionModel(nfeatures=NFEATURES, ntargets=1,
                               nlayers=n_layers, hidden_size=n_hidden, dropout=dropout)
    return model

model=load_untrained_model()

print(f'Training for {n_iterations} iterations')
start=time.time()
print('estimating %s\n' % target)
IQN_trace=([], [], [], [])
traces_step = 50
n_iterations=20000
IQN = run(model=model, target=target,train_x=train_x, train_t=train_t, 
        valid_x=valid_x, valid_t=valid_t, traces=IQN_trace, n_batch=2560, 
        n_iterations=n_iterations, traces_step=50, traces_window=50,
        save_model=False)

end=time.time()
difference=end-start
print('evaluating m took ',difference, 'seconds')


# In[150]:


def simple_eval(model):
    model.eval()
    valid_x_tensor=torch.from_numpy(valid_x).float()
    pred = IQN(valid_x_tensor)
    p = pred.detach().numpy()
    plt.hist(p, label='just L-scaled');plt.legend();plt.show()
    
simple_eval(IQN)


# In[156]:


print(f'spliting data for {target}')
train_t, train_x = normal_split_t_x(df= train_data, target = target, input_features=features)
print('train_t shape = ',train_t.shape , 'train_x shape = ', train_x.shape)
print('\n Training features:\n')
print(train_x)
valid_t, valid_x = normal_split_t_x(df= train_data, target = target, input_features=features)
print('valid_t shape = ',valid_t.shape , 'valid_x shape = ', valid_x.shape)

print('no need to train_test_split since we already have the split dataframes')
simple_eval(IQN)


# ### Evalute model and save evaluated data

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




