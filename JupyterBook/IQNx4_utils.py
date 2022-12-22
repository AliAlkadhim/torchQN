
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
from IPython.display import Image, display
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
    utils_dir = os.path.join(IQN_BASE, 'utils')
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
    get_ipython().run_cell(cell)
    
    
@debug
def get_model_params_simple():
    dropout=0.2
    n_layers = 2
    n_hidden=32
    starting_learning_rate=1e-3
    print('n_iterations, n_layers, n_hidden, starting_learning_rate, dropout')
    return n_iterations, n_layers, n_hidden, starting_learning_rate, dropout




# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

DATA_DIR=os.environ['DATA_DIR']
X       = ['genDatapT', 'genDataeta', 'genDataphi', 'genDatam', 'tau']

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

# In[20]:


all_variable_cols=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']
all_cols=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam', 'tau']




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
        
        

# def split_t_x(df, target, input_features):
#     """ Get teh target as the ratio, according to the T equation"""
    
#     if target=='RecoDatam':
#         t = T('m', scaled_df=train_data_m)
#     if target=='RecoDatapT':
#         t = T('pT', scaled_df=train_data_m)
#     if target=='RecoDataeta':
#         t = T('eta', scaled_df=train_data_m)
#     if target=='RecoDataphi':
#         t = T('phi', scaled_df=train_data_m)
#     x = np.array(df[input_features])
#     return np.array(t), x



# def apply_z_to_features():
#     """TO ensure this z scaling is only applied once to the training features, we use a generator """
#     for i in range(NFEATURES-1):
#         train_x[:,i] = z(train_x[:,i])
#         test_x[:,i] = z(test_x[:,i])
#         valid_x[:,i] = z(valid_x[:,i])
#     yield train_x 
#     yield test_x 
#     yield valid_x





# ### Apply $z$ to targets before training

def apply_z_to_targets():
    train_t_ratio_ = z(train_t_ratio) 
    test_t_ratio_ = z(test_t_ratio) 
    valid_t_ratio_ = z(valid_t_ratio)
    
    yield train_t_ratio_
    yield test_t_ratio_
    yield valid_t_ratio_
    

class RegularizedRegressionModel(nn.Module):
    """Used for hyperparameter tuning """
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
                #dropout only in the first layer
                #Dropout seems to worsen model performance
                layers.append(nn.Dropout(dropout))
                #ReLU activation 
                layers.append(nn.LeakyReLU())
            else:
                #if this is not the first layer (we dont have layers)
                layers.append(nn.Linear(hidden_size, hidden_size))
                # layers.append(nn.BatchNorm1d(hidden_size))
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
    
    
class TrainingRegularizedRegressionModel(nn.Module):
    """Used for training, and adds more regularization to prevent overfitting """
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
                layers.append(nn.Dropout(dropout))
                layers.append(nn.LeakyReLU())
                #output layer:
        layers.append(nn.Linear(hidden_size, ntargets)) 

        # only for classification add sigmoid
        # layers.append(nn.Sigmoid())
            #we have defined sequential model using the layers in oulist 
        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        return self.model(x)




# ## Hyperparameter Training Workflow


def get_tuning_sample():
    sample=int(200000)
    # train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample
    get_whole=True
    if get_whole:
        train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample = train_x, train_t_ratio, valid_x, valid_t_ratio
    else:
        train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample=train_x[:sample], train_t_ratio[:sample], valid_x[:sample], valid_t_ratio[:sample]
    return train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample



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

    # for epoch in range(EPOCHS):
    # train_loss = trainer.train(train_x_sample, train_t_ratio_sample)
        #test loss
    valid_loss=trainer.evaluate(valid_x_sample, valid_t_ratio_sample)

        # print(f"{epoch} \t {train_loss} \t {valid_loss}")
        
        # if valid_loss<best_loss:
        #     best_loss=valid_loss
        # else:
        #     early_stopping_coutner+=1
        # if early_stopping_coutner > early_stopping_iter:
            # break
            
    # return best_loss
    return valid_loss




def objective(trial):
    CLUSTER=False
    #cluster has greater memory than my laptop, which allows higher max values in hyperparam. search space
    if CLUSTER:
        nlayers_max,n_hidden_max, batch_size_max=int(24),int(350), int(1e5)
        n_trials=1000
    else:
        nlayers_max,n_hidden_max, batch_size_max=int(6),int(256), int(3e4)
        n_trials=2
    #hyperparameter search space:
    params = {
          "nlayers": trial.suggest_int("nlayers",1,nlayers_max),      
          "hidden_size": trial.suggest_int("hidden_size", 1, n_hidden_max),
          "dropout": trial.suggest_float("dropout", 0.0,0.5),
          "optimizer_name" : trial.suggest_categorical("optimizer_name", ["RMSprop", "SGD"]),
          "momentum": trial.suggest_float("momentum", 0.0,0.99),
          "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-2),
          "batch_size": trial.suggest_int("batch_size", 500, batch_size_max)

        }
    
    for step in range(10):

        temp_loss = run_train(params,save_model=False)
        trial.report(temp_loss, step)
        #activate pruning (early stopping if the current step in the trial has unpromising results)
        #instead of doing lots of iterations, do less iterations and more steps in each trial,  
        #such that a trial is terminated if a step yields an unpromising loss.
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return temp_loss

@time_type_of_func(tuning_or_training='tuning')
def tune_hyperparameters(save_best_params):
    

    sampler=False#use different sampling technique than the defualt one if sampler=True.
    if sampler:
        #choose a different sampling strategy (https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html#optuna.samplers.CmaEsSampler)
        # sampler=optuna.samplers.RandomSampler()
        study=optuna.create_study(direction='minimize',
                                  pruner=optuna.pruners.MedianPruner(), sampler=sampler)
    else:
        #but the default sampler is usually better - no need to change it!
        study=optuna.create_study(direction='minimize',
                                  pruner=optuna.pruners.HyperbandPruner())
    print(f'using {n_trials} trials for tuning')
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial
    print('best model parameters', best_trial.params)

    best_params=best_trial.params#this is a dictionary
    #save best hyperapameters in a pandas dataframe as a .csv
    if save_best_params:
        tuned_dir = os.path.join(IQN_BASE,'best_params')
        mkdir('tuned_dir')
        filename=os.path.join(tuned_dir,'best_params_mass_%s_trials.csv' % str(int(n_trials)))
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
    return study



def load_untrained_model():
    model=TrainingRegularizedRegressionModel(nfeatures=NFEATURES, ntargets=1,
                               nlayers=n_layers, hidden_size=hidden_size, dropout=dropout)
    
    