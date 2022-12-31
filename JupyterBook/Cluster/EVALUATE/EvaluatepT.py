
import numpy as np; import pandas as pd
# import scipy as sp; import scipy.stats as st
import torch; import torch.nn as nn; print(f"using torch version {torch.__version__}")
#use numba's just-in-time compiler to speed things up
# from numba import njit
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
import sys; import os
#from IPython.display import Image, display
# from importlib import import_module
#import plotly
try:
    import optuna
    print(f"using (optional) optuna version {optuna.__version__}")
except Exception:
    print('optuna is only used for hyperparameter tuning, not critical!')
    pass
import argparse
import time
# import sympy as sy
#import ipywidgets as wid; 

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
# seed = 128
# rnd  = np.random.RandomState(seed)
#sometimes jupyter doesnt initialize MathJax automatically for latex, so do this:

try:
    IQN_BASE = os.environ['IQN_BASE']
    print('BASE directoy properly set = ', IQN_BASE)
    utils_dir = os.path.join(IQN_BASE, 'utils/')
    sys.path.append(utils_dir)
    import utils
    #usually its not recommended to import everything from a module, but we know
    #whats in it so its fine
    from utils import *
    print('DATA directory also properly set, in %s' % os.environ['DATA_DIR'])
except Exception:
    # IQN_BASE=os.getcwd()
    print("""\nBASE directory not properly set. Read repo README.    If you need a function from utils, use the decorator below, or add utils to sys.path\n
    You can also do 
    os.environ['IQN_BASE']=<ABSOLUTE PATH FOR THE IQN REPO>
    or
    os.environ['IQN_BASE']=os.getcwd()""")
    pass

# from IQNx4_utils import *
IQN_BASE = os.environ['IQN_BASE']
print('BASE directoy properly set = ', IQN_BASE)
utils_dir = os.path.join(IQN_BASE, 'utils/')
sys.path.append(utils_dir)
#usually its not recommended to import everything from a module, but we know
#whats in it so its fine
from utils import *

#or use joblib for caching on disk
from joblib import  Memory
USE_BRADEN_SCALING=False

################################### CONFIGURATIONS ###################################
DATA_DIR=os.environ['DATA_DIR']
print(f'using DATA_DIR={DATA_DIR}')
JUPYTER=False
use_subsample=False
# use_subsample=True
if use_subsample:
    SUBSAMPLE=int(1e5)#subsample use for development - in production use whole dataset
else:
    SUBSAMPLE=None
    
memory = Memory(DATA_DIR)




###############################################################################################
y_label_dict ={'RecoDatapT':'$p(p_T)$'+' [ GeV'+'$^{-1} $'+']',
                    'RecoDataeta':'$p(\eta)$', 'RecoDataphi':'$p(\phi)$',
                    'RecoDatam':'$p(m)$'+' [ GeV'+'$^{-1} $'+']'}

loss_y_label_dict ={'RecoDatapT':'$p_T^{reco}$',
                    'RecoDataeta':'$\eta^{reco}$', 
                    'RecoDataphi':'$\phi^{reco}$',
                    'RecoDatam':'$m^{reco}$'}



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

               'RecoDataphi'  : {'inputs': ['RecoDatam', 'RecoDatapT', 'RecoDataeta']+X,
                               'xlabel': r'$\phi$' ,
                                'ylabel' :'$\phi^{reco}$',
                               'xmin'  : -3.2, 
                               'xmax'  :3.2}
              }


# Load and explore raw (unscaled) dataframes



all_variable_cols=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']
all_cols=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam', 'tau']


################################### Load unscaled dataframes ###################################
@memory.cache
def load_raw_data():
    print(f'\nSUBSAMPLE = {SUBSAMPLE}\n')
    raw_train_data=pd.read_csv(os.path.join(DATA_DIR,'train_data_10M_2.csv'),
                        usecols=all_cols,
                        nrows=SUBSAMPLE
                        )

    raw_test_data=pd.read_csv(os.path.join(DATA_DIR,'test_data_10M_2.csv'),
                        usecols=all_cols,
                        nrows=SUBSAMPLE
                        )

    raw_valid_data=pd.read_csv(os.path.join(DATA_DIR,'validation_data_10M_2.csv'),
                        usecols=all_cols,
                        nrows=SUBSAMPLE
                        )


    print('\n RAW TRAIN DATA\n')
    print(raw_train_data.shape)
    raw_train_data.describe()#unscaled
    print('\n RAW TEST DATA\n')
    print(raw_test_data.shape)
    raw_test_data.describe()#unscaled

    return raw_train_data, raw_test_data, raw_valid_data

raw_train_data, raw_test_data, raw_valid_data =load_raw_data()




################ Load scaled data##############
@time_type_of_func(tuning_or_training='loading')
@memory.cache
def load_scaled_dataframes():
    print("SCALED TRAIN DATA")
    scaled_train_data = pd.read_csv(
        os.path.join(DATA_DIR, "scaled_train_data_10M_2.csv"),
        usecols=all_cols,
        nrows=SUBSAMPLE,
    )

    print("TRAINING FEATURES\n", scaled_train_data.head())

    scaled_test_data = pd.read_csv(
        os.path.join(DATA_DIR, "scaled_test_data_10M_2.csv"),
        usecols=all_cols,
        nrows=SUBSAMPLE,
    )

    scaled_valid_data = pd.read_csv(
        os.path.join(DATA_DIR, "scaled_valid_data_10M_2.csv"),
        usecols=all_cols,
        nrows=SUBSAMPLE,
    )
    return scaled_train_data, scaled_test_data, scaled_valid_data

# print('\nTESTING FEATURES\n', scaled_test_data.head())

# print('\ntrain set shape:',  scaled_train_data.shape)
# print('\ntest set shape:  ', scaled_test_data.shape)
# # print('validation set shape:', valid_data.shape)

# scaled_train_data, scaled_test_data, scaled_valid_data = load_scaled_dataframes()

#######################################
target = 'RecoDatapT'
source  = FIELDS[target]
features= source['inputs']
print('Training Features:\n', features)
print('\nTarget = ', target)

print('USING NEW DATASET\n')

################################ SPLIT###########
# Currently need the split function again here

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

def split_t_x(df, target, input_features):
    """ Get teh target as the ratio, according to the T equation"""
    
    if target=='RecoDatam':
        t = T('m', scaled_df=scaled_train_data)
    if target=='RecoDatapT':
        t = T('pT', scaled_df=scaled_train_data)
    if target=='RecoDataeta':
        t = T('eta', scaled_df=scaled_train_data)
    if target=='RecoDataphi':
        t = T('phi', scaled_df=scaled_train_data)
    x = np.array(df[input_features])
    return np.array(t), x

def split_t_x_test(df, target, input_features):
    """ Get teh target as the ratio, according to the T equation"""
    
    if target=='RecoDatam':
        t = T('m', scaled_df=scaled_test_data)
    if target=='RecoDatapT':
        t = T('pT', scaled_df=scaled_test_data)
    if target=='RecoDataeta':
        t = T('eta', scaled_df=scaled_test_data)
    if target=='RecoDataphi':
        t = T('phi', scaled_df=scaled_test_data)
    x = np.array(df[input_features])
    return np.array(t), x


#########################################################################
@memory.cache
def normal_split_t_x(df, target, input_features):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    # t = np.array(df[target])
    t = np.array(df[target])
    x = np.array(df[input_features])
    return t, x


if USE_BRADEN_SCALING:
    print(f"spliting data for {target}")
    train_t, train_x = split_t_x(
        df=scaled_train_data, target=target, input_features=features
    )
    print("train_t shape = ", train_t.shape, "train_x shape = ", train_x.shape)
    print("\n Training features:\n")
    print(train_x)
    valid_t, valid_x = split_t_x(
        df=scaled_valid_data, target=target, input_features=features
    )
    print("valid_t shape = ", valid_t.shape, "valid_x shape = ", valid_x.shape)
    test_t, test_x = split_t_x(df=scaled_test_data, target=target, input_features=features)
    print("test_t shape = ", test_t.shape, "test_x shape = ", test_x.shape)

else:
    print(f"spliting data for {target}")
    train_t, train_x = normal_split_t_x(
    df=raw_train_data, target=target, input_features=features
    )
    print("train_t shape = ", train_t.shape, "train_x shape = ", train_x.shape)
    print("\n Training features:\n")
    print(train_x)
    valid_t, valid_x = normal_split_t_x(
    df=raw_valid_data, target=target, input_features=features
    )
    print("valid_t shape = ", valid_t.shape, "valid_x shape = ", valid_x.shape)
    test_t, test_x = normal_split_t_x(df=raw_test_data, target=target, input_features=features)
    print("test_t shape = ", test_t.shape, "test_x shape = ", test_x.shape)




print("no need to train_test_split since we already have the split dataframes")
print(valid_x.mean(axis=0), valid_x.std(axis=0))
print(train_x.mean(axis=0), train_x.std(axis=0))
print(valid_t.mean(), valid_t.std())
print(train_t.mean(), train_t.std())

################ Apply Z scaling############
def z(x):
    eps=1e-20
    return (x - np.mean(x))/(np.std(x)+ eps)
def z_inverse(xprime, x):
    return xprime * np.std(x) + np.mean(x)

def z2(x, mean, std):
    """
    Args:
        x ([type]): [description]
        mean ([type]): [description]
        std ([type]): [description]

    Returns:
        [type]: [description]
    """
    eps=1e-20
    scaled = (x - mean)/(std+ eps)
    return np.array(scaled, dtype=np.float64)

def z_inverse(xprime, x):
    unscaled=xprime * np.std(x) + np.mean(x)
    return np.array(unscaled, dtype=np.float64)

def z_inverse2(xprime, train_mean, train_std):
    """mean original train mean, std: original. Probably not needed  """
    return xprime * train_std + train_mean

def apply_z_to_features(TRAIN_SCALE_DICT, train_x, test_x, valid_x):
    """TO ensure this z scaling is only applied once to the training features, we use a generator. 
    This doesn't change the shapes of anything, just applies z to all the feature columns other than tau"""
    NFEATURES=train_x.shape[1]
    for i in range(NFEATURES-1):
        variable = list(TRAIN_SCALE_DICT)[i]
        train_mean = float(TRAIN_SCALE_DICT[variable]['mean'])
        train_std = float(TRAIN_SCALE_DICT[variable]['std'])
        train_x[:,i] = z2(train_x[:,i], mean = train_mean, std=train_std)
        test_x[:,i] = z2(test_x[:,i], mean = train_mean, std=train_std)
        valid_x[:,i] = z2(valid_x[:,i], mean = train_mean, std=train_std)
    yield train_x 
    yield test_x 
    yield valid_x


def apply_z_to_targets(train_t, test_t, valid_t):
    train_mean = np.mean(train_t)
    train_std = np.std(train_t)
    train_t_ = z2(train_t, mean = train_mean, std = train_std) 
    test_t_ = z2(test_t, mean = train_mean, std = train_std) 
    valid_t_ = z2(valid_t, mean = train_mean, std = train_std)
    
    yield train_t_
    yield test_t_
    yield valid_t_
    
@debug
def save_model(model, PATH):
    print(model)
    torch.save(model.state_dict(), PATH)
    print('\ntrained model dictionary saved in %s' % PATH)

@debug
def load_model(PATH, PARAMS):
    # n_layers = int(BEST_PARAMS["n_layers"]) 
    # hidden_size = int(BEST_PARAMS["hidden_size"])
    # dropout = float(BEST_PARAMS["dropout"])
    # optimizer_name = BEST_PARAMS["optimizer_name"].to_string().split()[1]
    # learning_rate =  float(BEST_PARAMS["learning_rate"])
    # batch_size = int(BEST_PARAMS["batch_size"])
    model=RegularizedRegressionModel(nfeatures=NFEATURES, ntargets=1,
                               nlayers=PARAMS['n_layers'], hidden_size=PARAMS['hidden_size'], dropout_1=PARAMS['dropout_1'], dropout_2=PARAMS['dropout_2'],
                               activation=PARAMS['activation'])
    model.load_state_dict(torch.load(PATH) )
    #OR
    #model=torch.load(PATH)#BUT HERE IT WILL BE A DICT (CANT BE EVALUATED RIGHT AWAY) DISCOURAGED! Also, use dictionary ".pth" which has both the model state dict and the PARAMS dict
    model.eval()
    print(model)
    return model

def simple_eval(model, test_x_z_scaled):
    model.eval()
    #evaluate on the scaled features
    valid_x_tensor=torch.from_numpy(test_x_z_scaled).float()
    # valid_x_tensor=torch.from_numpy(train_x).float()
    pred = model(valid_x_tensor)
    p = pred.detach().numpy()
    # if USE_BRADEN_SCALING:
    #     fig, ax = plt.subplots(1,1)
    #     label=FIELDS[target]['ylabel']
    #     ax.hist(p, label=f'Predicted post-z ratio for {label}', alpha=0.4, density=True)
    #     # orig_ratio = z(T('m', scaled_df=scaled_train_data))
    #     orig_ratio = z(T('m', scaled_df=scaled_test_data))
    #     print(orig_ratio[:5])
    #     ax.hist(orig_ratio, label = f'original post-z ratio for {label}', alpha=0.4,density=True)
    #     ax.grid()
    #     set_axes(ax, xlabel='predicted $T$')
    # print('predicted ratio shape: ', p.shape)
    return p
    
    
# def main():
n_iterations=int(1e4)



PARAMS_ = {
    "n_layers": int(8),
    "hidden_size": int(5),
    "dropout_1": float(0.6),
    "dropout_2": float(0.1),
    "activation": "LeakyReLU"
    #   'optimizer_name':'SGD',
}

if USE_BRADEN_SCALING:
    TRAIN_SCALE_DICT = get_scaling_info(scaled_train_data)
    print(TRAIN_SCALE_DICT)
    print("\n\n")
    # TEST_SCALE_DICT = get_scaling_info(scaled_test_data)
    # print(TEST_SCALE_DICT)
else:
    TRAIN_SCALE_DICT = get_scaling_info(raw_train_data)
    print(TRAIN_SCALE_DICT)
    print("\n\n")
    # TEST_SCALE_DICT = get_scaling_info(scaled_test_data)
    # print(TEST_SCALE_DICT)
    
NFEATURES=train_x.shape[1]
# to features
apply_z_generator = apply_z_to_features(TRAIN_SCALE_DICT, train_x, test_x, valid_x)
train_x_z_scaled = next(apply_z_generator)
test_x_z_scaled = next(apply_z_generator)
valid_x_z_scaled = next(apply_z_generator)
print(valid_x_z_scaled.mean(axis=0), valid_x_z_scaled.std(axis=0))
print(train_x_z_scaled.mean(axis=0), train_x_z_scaled.std(axis=0))
#to targets
apply_z_to_targets_generator = apply_z_to_targets(train_t, test_t, valid_t)
train_t_z_scaled = next(apply_z_to_targets_generator)
test_t_z_scaled = next(apply_z_to_targets_generator)
valid_t_z_scaled = next(apply_z_to_targets_generator)
print(valid_t_z_scaled.mean(), valid_t_z_scaled.std())
print(train_t_z_scaled.mean(), train_t_z_scaled.std())


# filename='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
filename ="Trained_IQNx4_%s_TUNED_0lin_with_high_noise.dict" % target

# 'Trained_IQNx4_%s_TUNED.dict' % target

# filename='Trained_IQNx4_RecoDatam_10K_iter.dict'
# print(f'model file name: {filename}')
trained_models_dir='trained_models'
mkdir(trained_models_dir)
PATH = os.path.join(IQN_BASE,'JupyterBook', 'Cluster', 'TRAIN', trained_models_dir , filename)
IQN_m = load_model(PATH, PARAMS_)

p = simple_eval(IQN_m, test_x_z_scaled)
# plt.show()
    
# if __name__=='__main__':
#     main()
def z_inverse(xprime, mean, std):
    return xprime * std + mean


range_=[20,80]
bins=50
REAL_RAW_DATA=raw_test_data

YLIM=(0.8,1.2)
###########GET REAL DIST###########
REAL_RAW_DATA = REAL_RAW_DATA[['RecoDatapT','RecoDataeta','RecoDataphi','RecoDatam']]
REAL_RAW_DATA.columns = ['realpT','realeta','realphi','realm']
REAL_DIST=REAL_RAW_DATA['realpT']
norm_data=REAL_RAW_DATA.shape[0]
#############GET EVALUATION DIST#############

raw_test_data.describe()
# m_reco = raw_test_data['RecoDatam']
# m_gen = raw_test_data['genDatam']
# plt.hist(m_reco,label=r'$m_{gen}^{test \ data}$');plt.legend();plt.show()

if USE_BRADEN_SCALING:
    orig_ratio = T('m', scaled_df=scaled_train_data)
    z_inv_f =z_inverse(xprime=p, mean=np.mean(orig_ratio), std=np.std(orig_ratio))
    L_obs = L(orig_observable=m_gen, label='m')
    z_inv_f = z_inv_f.flatten();print(z_inv_f.shape)

    factor = (z_inv_f * (L_obs  + 10) )-10
    m_pred = L_inverse(L_observable=factor, label='m')

else:
    m_pred =  z_inverse2(xprime = p, train_mean = TRAIN_SCALE_DICT[target]['mean']
                         , train_std=TRAIN_SCALE_DICT[target]['std'])
    m_pred=m_pred.flatten()
    
# eval_data=pd.read_csv(DATA_DIR+'/test_data_10M_2.csv')
eval_data=pd.read_csv('AUTOREGRESSIVE_m_Prime.csv')
ev_features=features #['RecoDatam']+X
eval_data=eval_data[ev_features]
eval_data['RecoDatapT']=m_pred
#save new distribution (m) in the eval data as autoregressive eval for next IQN
new_cols=['RecoDatam', 'RecoDatapT']+X
eval_data=eval_data.reindex(columns=new_cols)
print('EVALUATION DATA NEW INDEX\n', eval_data.head())

eval_data.to_csv(os.path.join(IQN_BASE,'JupyterBook', 'Cluster','EVALUATE', 'AUTOREGRESSIVE_m_Prime_pT_Prime.csv'))

AUTOREGRESSIVE_DIST=pd.read_csv(os.path.join(IQN_BASE,'JupyterBook', 'Cluster','EVALUATE', 'AUTOREGRESSIVE_m_Prime_pT_Prime.csv'))


# norm_IQN=AUTOREGRESSIVE_DIST.shape[0]

norm_autoregressive=AUTOREGRESSIVE_DIST.shape[0]
norm_IQN=norm_autoregressive
print('norm_data',norm_data,'\nnorm IQN',norm_IQN,'\nnorm_autoregressive', norm_autoregressive)

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


real_label_counts_m, predicted_label_counts_m, label_edges_m = get_hist_simple('m')

def plot_one_m():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
    ax1.step(label_edges_m, real_label_counts_m/norm_data, where="mid", color="k", linewidth=0.5)# step real_count_pt
    ax1.step(label_edges_m, predicted_label_counts_m/norm_IQN, where="mid", color="#D7301F", linewidth=0.5)# step predicted_count_pt
    ax1.scatter(label_edges_m, real_label_counts_m/norm_data, label="reco",  color="k",facecolors='none', marker="o", s=5, linewidth=0.5)
    ax1.scatter(label_edges_m,predicted_label_counts_m/norm_IQN, label="predicted", color="#D7301F", marker="x", s=5, linewidth=0.5)
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
    # ax2.set_ylab el(r"$\\frac{\\textnormal{predicted}}{\\textnormal{reco}}$")
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
    
plot_one_m()