import os, sys

# the standard module for tabular data
import pandas as pd

# the standard module for array manipulation
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt

# standard scientific python module
import scipy as sp
import scipy.stats as st
import scipy.optimize as op

# standard symbolic algebra module
# import sympy as sm
# sm.init_printing()

# module to save results
# import joblib as jb

# pytorch
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split
# linearly transform a feature to zero mean and unit variance
from sklearn.preprocessing import StandardScaler

# to reload modules
import importlib
from torch.utils.data import Dataset

device = torch.device("cuda:0")

# update fonts
FONTSIZE = 18
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

####################################################################



class CustomDataset(Dataset):
    """This takes the index for the data and target and gives dictionary of tensors of data and targets.
    For example we could do train_dataset = CustomDataset(train_data, train_targets); test_dataset = CustomDataset(test_data, test_targets)
 where train and test_dataset are np arrays that are reshaped to (-1,1).
 Then train_dataset[0] gives a dictionary of samples "X" and targets"""
    def __init__(self, data, targets):
        self.data = data
        self.targets=targets
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        
        current_sample = self.data[idx, :]
        current_target = self.targets[idx]
        return {"x": torch.tensor(current_sample, dtype = torch.float),
               "y": torch.tensor(current_target, dtype= torch.float),
               }#this already makes the targets made of one tensor (of one value) each


def split_t_x(df, target, source, scalers):
    # change from pandas dataframe format to a numpy array
    scaler_t, scaler_x = scalers
    t = np.array(scaler_t.transform(df[target].to_numpy().reshape(-1, 1)))
    #where scaler_t is a StandardScaler() object, which has the .transorm method
    x = np.array(scaler_x.transform(df[source]))
    t = t.reshape(-1,)
    return t, x

class RegressionModel(nn.Module):
    #inherit from the super class
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size):
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
                # layers.append(nn.ReLU())
                layers.append(nn.LeakyReLU())
            else:
                #if this is not the first layer (we dont have layers)
                layers.append(nn.Linear(hidden_size, hidden_size))
                # layers.append(nn.BatchNorm1d(hidden_size))
                #Dropout seems to worsen model performance
                # layers.append(nn.Dropout(dropout))
                # layers.append(nn.ReLU())
                layers.append(nn.LeakyReLU())
                #output layer:
        layers.append(nn.Linear(hidden_size, ntargets)) 
        
        # ONLY IF ITS A CLASSIFICATION, ADD SIGMOID
        #layers.append(nn.Sigmoid())
            #we have defined sequential model using the layers in oulist 
        self.model = nn.Sequential(*layers)
            
    
    def forward(self, x):
        return self.model(x)

class RegularizedRegressionModel(nn.Module):
    #inherit from the super class
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size):
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
                #ReLU activation 
                layers.append(nn.LeakyReLU())
            else:
                #if this is not the first layer (we dont have layers)
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                #Dropout seems to worsen model performance
                layers.append(nn.LeakyReLU())
                #output layer:
        layers.append(nn.Linear(hidden_size, ntargets)) 
        
        # ONLY IF ITS A CLASSIFICATION, ADD SIGMOID
        #layers.append(nn.Sigmoid())
            #we have defined sequential model using the layers in oulist 
        self.model = nn.Sequential(*layers)
            
    
    def forward(self, x):
        return self.model(x)





class RegressionEngine:
    """loss, training and evaluation"""
    def __init__(self, model, optimizer):
                 #, device):
        self.model = model
        #self.device= device
        self.optimizer = optimizer
        
    #the loss function returns the loss function. It is a static method so it doesn't need self
    @staticmethod
    def quadratic_loss(targets, outputs):
         return nn.MSELoss()(outputs, targets)

    @staticmethod
    def average_quadratic_loss(targets, outputs):
    # f and t must be of the same shape
        return  torch.mean((outputs - targets)**2)
    
    @staticmethod
    def average_absolute_error(targets, outputs):
    # f and t must be of the same shape
        return  torch.mean(abs(outputs - targets))
    
    
    @staticmethod
    def average_cross_entropy_loss(targets, outputs):
        # f and t must be of the same shape
        loss = torch.where(targets > 0.5, torch.log(outputs), torch.log(1 - outputs))
        # the above means loss = log outputs, if target>0.5, and log(1-output) otherwise
        return -torch.mean(loss)
    
    @staticmethod
    def average_quantile_loss(targets, outputs):
        # f and t must be of the same shape
        tau = torch.rand(outputs.shape)
        return torch.mean(torch.where(targets >= outputs, 
                                      tau * (targets - outputs), 
                                      (1 - tau)*(outputs - targets)))


    def train(self, data_loader):
        """the training function: takes the training dataloader"""
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()#only optimize weights for the current batch, otherwise it's meaningless!
            inputs = data["x"]
            targets = data["y"]
            outputs = self.model(inputs)
            loss = self.average_quantile_loss(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    
    def evaluate(self, data_loader):
        """the training function: takes the training dataloader"""
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data["x"]#.to(self.device)
            targets = data["y"]#.to(self.device)
            outputs = self.model(inputs)
            loss = self.average_quantile_loss(targets, outputs)
            final_loss += loss.item()
        return final_loss / len(data_loader)



class ModelHandler:
    def __init__(self, model, scalers):
        self.model  = model
        self.scaler_t, self.scaler_x = scalers
        
        self.scale  = self.scaler_t.scale_[0] # for output
        self.mean   = self.scaler_t.mean_[0]  # for output
        self.fields = self.scaler_x.feature_names_in_
        
    def __call__(self, df):
        
        # scale input data
        x  = np.array(self.scaler_x.transform(df[self.fields]))
        x  = torch.Tensor(x)

        # go to evaluation mode
        self.model.eval()
    
        # compute,reshape to a 1d array, and convert to a numpy array
        Y  = self.model(x).view(-1, ).detach().numpy()
        
        # rescale output
        Y  = self.mean + self.scale * Y
        
        if len(Y) == 1:
            return Y[0]
        else:
            return Y
        
    def show(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                print()


N_CNN_KERNEL=2
NFEATURES=1#train_x.shape[1]
N_MULT_FACTOR=2
N_HIDDEN=NFEATURES * N_MULT_FACTOR

class CNN_MODEL(nn.Module):    
    def __init__(self, n_feature, n_hidden, n_output, n_cnn_kernel, n_mult_factor=N_MULT_FACTOR):
        super(CNN_MODEL, self).__init__()
        self.n_feature=n_feature
        self.n_hidden=n_hidden
        self.n_output= n_output 
        self.n_cnn_kernel=n_cnn_kernel
        self.n_mult_factor=n_mult_factor
        self.n_l2_hidden=self.n_hidden * (self.n_mult_factor - self.n_cnn_kernel + 3)
        self.n_out_hidden=int (self.n_l2_hidden/2)
                        
        self.l1 = nn.Sequential(
            torch.nn.Linear(self.n_feature, self.n_hidden),
            torch.nn.Dropout(p=1 -.85),            
            torch.nn.LeakyReLU (0.1),            
            torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True)            
        )                
        self.c1= nn.Sequential(            
            torch.nn.Conv1d(self.n_feature, self.n_hidden, 
                            kernel_size=(self.n_cnn_kernel,), stride=(1,), padding=(1,)),
            torch.nn.Dropout(p=1 -.75),            
            torch.nn.LeakyReLU (0.1),
            torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True)        
        )                        
        self.out = nn.Sequential(
            torch.nn.Linear(self.n_l2_hidden,
                            self.n_output),  
        )                

        
    def forward(self, x):
        varSize=x.shape[0] # must be calculated here in forward() since its is a dynamic size        
        x=self.l1(x)                
        # for CNN        
        x = x.view(varSize,self.n_feature,self.n_mult_factor)
        x=self.c1(x)
        # for Linear layer
        x = x.view(varSize, self.n_hidden * (self.n_mult_factor -self.n_cnn_kernel + 3))
#         x=self.l2(x)                    
        x=self.out(x)   
        return x

# model = CNN_MODEL(n_feature=NFEATURES, n_hidden=N_HIDDEN, n_output=1, n_cnn_kernel=N_CNN_KERNEL)   # define the network    



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