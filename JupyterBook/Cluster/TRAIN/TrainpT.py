#!/usr/bin/env python
# coding: utf-8



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

# def show_jupyter_image(image_filename, width = 1300, height = 300):
#     """Show a saved image directly in jupyter. Make sure image_filename is in your IQN_BASE !"""
#     display(Image(os.path.join(IQN_BASE,image_filename), width = width, height = height  ))
    
    
# def use_svg_display():
#     """Use the svg format to display a plot in Jupyter (better quality)"""
#     from matplotlib_inline import backend_inline
#     backend_inline.set_matplotlib_formats('svg')

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
        
# from IPython.core.magic import register_cell_magic

# @register_cell_magic
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
    #get_ipython().run_cell(cell)
    
    
# @debug
# def get_model_params_simple():
#     dropout=0.2
#     n_layers = 2
#     n_hidden=32
#     starting_learning_rate=1e-3
#     print('n_iterations, n_layers, n_hidden, starting_learning_rate, dropout')
#     return n_iterations, n_layers, n_hidden, starting_learning_rate, dropout




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



################################### CONFIGURATIONS ###################################
DATA_DIR=os.environ['DATA_DIR']
print(f'using DATA_DIR={DATA_DIR}')
JUPYTER=False
use_subsample=False
# use_subsample=True
if use_subsample:
    SUBSAMPLE=int(1e4)#subsample use for development - in production use whole dataset
else:
    SUBSAMPLE=None
    




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
print(f'SUBSAMPLE = {SUBSAMPLE}')
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
print('\n RAW TRAIN DATA\n')
print(raw_train_data.shape)
raw_train_data.describe()#unscaled
print('\n RAW TEST DATA\n')
print(raw_test_data.shape)
raw_test_data.describe()#unscaled

########## Generate scaled data###############
# scaled_train_data = L_scale_df(raw_train_data, title='scaled_train_data_10M_2.csv',
#                              save=True)
# print('\n\n')
# scaled_test_data = L_scale_df(raw_test_data,  title='scaled_test_data_10M_2.csv',
#                             save=True)
# print('\n\n')

# scaled_valid_data = L_scale_df(raw_valid_data,  title='scaled_valid_data_10M_2.csv',
#                             save=True)

# explore_data(df=scaled_train_data, title='Braden Kronheim-L-scaled Dataframe', scaled=True)

################ Load scaled data##############
print('SCALED TRAIN DATA')
train_data_m=pd.read_csv(os.path.join(DATA_DIR,'scaled_train_data_10M_2.csv'),
                       usecols=all_cols,
                       nrows=SUBSAMPLE)

print('TRAINING FEATURES\n', train_data_m.head())

test_data_m= pd.read_csv(os.path.join(DATA_DIR,'scaled_test_data_10M_2.csv'),
                       usecols=all_cols,
                       nrows=SUBSAMPLE)

valid_data_m= pd.read_csv(os.path.join(DATA_DIR,'scaled_valid_data_10M_2.csv'),
                       usecols=all_cols,
                       nrows=SUBSAMPLE)

# print('\nTESTING FEATURES\n', test_data_m.head())

# print('\ntrain set shape:',  train_data_m.shape)
# print('\ntest set shape:  ', test_data_m.shape)
# # print('validation set shape:', valid_data.shape)

scaled_train_data = train_data_m
scaled_test_data = test_data_m
scaled_valid_data = valid_data_m

TRAIN_SCALE_DICT = get_scaling_info(scaled_train_data);print(TRAIN_SCALE_DICT)
print('\n\n')
TEST_SCALE_DICT = get_scaling_info(scaled_test_data);print(TEST_SCALE_DICT)

#######################################
target = 'RecoDatapT'
source  = FIELDS[target]
features= source['inputs']
print('Training Features:\n', features)
print('\nTarget = ', target)

print('USING NEW DATASET\n')

################################ SPLIT###########
# Currently need the split function again here
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


print(f'spliting data for {target}')
train_t_ratio, train_x = split_t_x(df= train_data_m, target = target, input_features=features)
print('train_t shape = ',train_t_ratio.shape , 'train_x shape = ', train_x.shape)
print('\n Training features:\n')
print(train_x)
valid_t_ratio, valid_x = split_t_x(df= valid_data_m, target = target, input_features=features)
print('valid_t shape = ',valid_t_ratio.shape , 'valid_x shape = ', valid_x.shape)
test_t_ratio, test_x = split_t_x(df= test_data_m, target = target, input_features=features)
print('test_t shape = ',test_t_ratio.shape , 'test_x shape = ', test_x.shape)
print('no need to train_test_split since we already have the split dataframes')
print(valid_x.mean(axis=0), valid_x.std(axis=0))
print(train_x.mean(axis=0), train_x.std(axis=0))
print(valid_t_ratio.mean(), valid_t_ratio.std())
print(train_t_ratio.mean(), train_t_ratio.std())
NFEATURES=train_x.shape[1]

################ Apply Z scaling############
def z(x):
    """used for targets """
    eps=1e-20
    return (x - np.mean(x))/(np.std(x)+ eps)

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
    return (x - mean)/(std+ eps)
def z_inverse(xprime, x):
    return xprime * np.std(x) + np.mean(x)

def z_inverse2(xprime, train_mean, train_std):
    """mean original train mean, std: original. Probably not needed  """
    return xprime * train_mean + train_std

def apply_z_to_features(TRAIN_SCALE_DICT, train_x, test_x, valid_x):
    """TO ensure this z scaling is only applied once to the training features, we use a generator. 
    This doesn't change the shapes of anything, just applies z to all the feature columns other than tau"""
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
    
    

def apply_z_to_targets(train_t_ratio, test_t_ratio, valid_t_ratio):
    train_mean = np.mean(train_t_ratio)
    train_std = np.std(train_t_ratio)
    train_t_ratio_ = z2(train_t_ratio, mean = train_mean, std = train_std) 
    test_t_ratio_ = z2(test_t_ratio, mean = train_mean, std = train_std) 
    valid_t_ratio_ = z2(valid_t_ratio, mean = train_mean, std = train_std)
    
    yield train_t_ratio_
    yield test_t_ratio_
    yield valid_t_ratio_
    
# to features
apply_z_generator = apply_z_to_features(TRAIN_SCALE_DICT, train_x, test_x, valid_x)
train_x = next(apply_z_generator)
test_x = next(apply_z_generator)
valid_x = next(apply_z_generator)
print(valid_x.mean(axis=0), valid_x.std(axis=0))
print(train_x.mean(axis=0), train_x.std(axis=0))

#to targets
apply_z_to_targets_generator = apply_z_to_targets(train_t_ratio, test_t_ratio, valid_t_ratio)
train_t_ratio = next(apply_z_to_targets_generator)
test_t_ratio = next(apply_z_to_targets_generator)
valid_t_ratio = next(apply_z_to_targets_generator)
print(valid_t_ratio.mean(), valid_t_ratio.std())
print(train_t_ratio.mean(), train_t_ratio.std())


#check that it looks correct
# fig = plt.figure(figsize=(10, 4))
# ax = fig.add_subplot(autoscale_on=True)
# ax.grid()
# for i in range(NFEATURES):
#     ax.hist(train_x[:,i], alpha=0.35, label=f'feature {i}' )
#     # set_axes(ax=ax, xlabel="Transformed features X' ",title="training features post-z score: X'=z(L(X))")
# ax.legend()
# plt.show()


######### Get beset hyperparameters
#tuned_dir = os.path.join(IQN_BASE,'best_params')
#tuned_filename=os.path.join(tuned_dir,'best_params_mass_%s_trials.csv' % str(int(n_trials)))
# BEST_PARAMS = pd.read_csv(os.path.join(IQN_BASE, 'best_params','best_params_Test_Trials.csv'))
#BEST_PARAMS=pd.read_csv(tuned_filename)
#print(BEST_PARAMS)


dropout = 0.25 #float(BEST_PARAMS["dropout"])s
optimizer_name ='Adam' #BEST_PARAMS["optimizer_name"]
print(type(optimizer_name))
#optimizer_name = BEST_PARAMS["optimizer_name"].to_string().split()[1]
def load_untrained_model(PARAMS):
    model=RegularizedRegressionModel(nfeatures=NFEATURES, ntargets=1,
                               nlayers=PARAMS['n_layers'], hidden_size=PARAMS['hidden_size'], dropout_1=PARAMS['dropout_1'], dropout_2=PARAMS['dropout_2'],
                               activation=PARAMS['activation'])
    # model.apply(initialize_weights)
    print(model)
    return model

# optimizer_name =  'Adam'
best_learning_rate =  3e-03 #float(BEST_PARAMS["learning_rate"])
momentum=0.3 #float(BEST_PARAMS["momentum"]) 
# best_optimizer_temp = getattr(torch.optim, optimizer_name)(model.parameters(), lr=best_learning_rate,
#                                                            momentum=momentum,
#                                                           amsgrad=True  )
batch_size = int(512 * 3) #512 #int(BEST_PARAMS["batch_size"])
# BATCHSIZE=10000
BATCHSIZE=batch_size 
# n_iterations=int(1e7)
n_iterations=int(1e3)


class SaveModelCheckpoint:
    def __init__(self, best_valid_loss=np.inf):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, model, current_valid_loss):
        if current_valid_loss < self.best_valid_loss:
            #update the best loss
            self.best_valid_loss = current_valid_loss
            filename_model='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
            #note that n_iterations is the total n_iterations, we dont want to save a million files for each iteration
            trained_models_dir='trained_models'
            mkdir(trained_models_dir)
            # on cluster, Im using another TRAIN directory
            PATH_model = os.path.join(IQN_BASE,'JupyterBook', 'Cluster', 'TRAIN', trained_models_dir , filename_model)
            torch.save(model.state_dict(), PATH_model)
            print(f"\nCurrent valid loss: {current_valid_loss};  saved better model at {PATH_model}")
            #save using .pth object which if a dictionary of dicionaries, so that I can have PARAMS saved in the same file
def train(model, optimizer, avloss, getbatch,
          train_x, train_t, 
          valid_x, valid_t,
          batch_size, 
          n_iterations, traces, 
          step, window):
    
    # to keep track of average losses
    xx, yy_t, yy_v, yy_v_avg = traces
    model_checkpoint=SaveModelCheckpoint()
    n = len(valid_x)
    
    print('Iteration vs average loss')
    print("%10s\t%10s\t%10s" %           ('iteration', 'train-set', 'test-set'))
    
    for ii in range(n_iterations):

        # set mode to training so that training specific 
        # operations such as dropout are enabled.
        # time_p_start = time.perf_counter()
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
        
        #add early stopping when the model starts to overfit, as determined by performance on
        #valid set (when valid loss plateaus or starts increasing when train loss keeps decreasing)
        if ii % step == 0:
            
            
            acc_t = validate(model, avloss, train_x[:n], train_t[:n]) 
            acc_v = validate(model, avloss, valid_x[:n], valid_t[:n])

             
            yy_t.append(acc_t)
            yy_v.append(acc_v)
            #save better models based on valid loss
            model_checkpoint(model=model, current_valid_loss=acc_v)
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
                print("%10d\t%10.6f\t%10.6f" %                       (xx[-1], yy_t[-1], yy_v[-1]))
            else:
                xx.append(xx[-1] + step)
                    
                print("\r%10d\t%10.6f\t%10.6f\t%10.6f" %                           (xx[-1], yy_t[-1], yy_v[-1], yy_v_avg[-1]), 
                      end='')
        # time_p_end = time.perf_counter()
        # time_for_this_iter = time_p_end-time_p_start
        # time_per_example = time_for_this_iter/batch_size
        # print(f'training time for one example: {time_per_example}')

    print()      
    return (xx, yy_t, yy_v, yy_v_avg)


@time_type_of_func(tuning_or_training='training')
def run(model, 
        train_x, train_t, 
        valid_x, valid_t, traces,
        n_batch, 
        n_iterations, 
        traces_step, 
        traces_window,
        save_model):

    learning_rate= best_learning_rate
    #add weight decay (important regularization to reduce overfitting)
    L2=1e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate,     
                                                     amsgrad=True, eps=1e-7,
                                                    #  momentum=momentum, 
                                                     weight_decay=L2
                                                     )
    
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
    # optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate, 
    #                                                  amsgrad=True,
    #                                                 #  momentum=momentum,
    #                                                 # weight_decay=L2
    #                                                  )
    # 10^-4
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


    # learning_rate=learning_rate/10
    # optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate, 
    #                                                  amsgrad=True,
    #                                                 #  momentum=momentum,
    #                                                 # weight_decay=L2
    #                                                  )
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
    
    # learning_rate=learning_rate/10
    # optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate, 
    #                                                  amsgrad=True,
    #                                                 #  momentum=momentum,
    #                                                 # weight_decay=L2
    #                                                  )
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
    

    # plot_average_loss(traces, n_iterations,target)

    if save_model:
        filename='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
        PATH = os.path.join(IQN_BASE, 'trained_models', filename)
        torch.save(model.state_dict(), PATH)
        print('\ntrained model dictionary saved in %s' % PATH)
    return  model


# ## See if trainig works on T ratio


@debug
def save_model(model, PATH):
    print(model)
    torch.save(model.state_dict(), PATH)
    print('\ntrained model dictionary saved in %s' % PATH)

@debug
def save_model_params(model, PATH):
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

# def main():
    #N_epochs X N_train_examples = N_iterations X batch_size
N_epochs = (n_iterations * BATCHSIZE)/int(train_x.shape[0])
print(f'training for {n_iterations} iteration, which is  {N_epochs} epochs')


PARAMS_ = {'n_layers':int(1), 'hidden_size':int(128), 'dropout_1':float(0.95), 'dropout_2':float(0.95), 'activation':'LeakyReLU'
        #   'optimizer_name':'SGD', 
          }
filename_model='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
trained_models_dir='trained_models'
mkdir(trained_models_dir)
# on cluster, Im using another TRAIN directory
PATH_model = os.path.join(IQN_BASE,'JupyterBook', 'Cluster', 'TRAIN', trained_models_dir , filename_model)

#to load untrained model (start training from scratch), uncomment the next line
# model=load_untrained_model(PARAMS_)
IQN_trace=([], [], [], [])
traces_step = 2
traces_window=traces_step
#to continune training of model (pickup where the previous training left off), uncomment below
model_trained=load_trained_model(PATH=PATH_model, PARAMS=PARAMS_)
IQN = run(model=model_trained,train_x=train_x, train_t=train_t_ratio, 
        valid_x=test_x, valid_t=test_t_ratio, traces=IQN_trace, n_batch=BATCHSIZE, 
        n_iterations=n_iterations, traces_step=traces_step, traces_window=traces_window,
        save_model=False)
