
import utils
import os, sys
import time
import json

# the standard module for tabular data
import pandas as pd

# the standard module for array manipulation
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt




# pytorch
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split
# linearly transform a feature to zero mean and unit variance
from sklearn.preprocessing import StandardScaler

# to reload modules
import importlib

import argparse

DATA_DIR=os.environ['DATA_DIR']

parser=argparse.ArgumentParser(description='train for different targets')
parser.add_argument('--N', type=str, help='size of the dataset you want to use. Options are 10M and 100K and 10M_2, the default is 10M_2', required=False,default='10M_2')
#N_epochs X N_train_examples = N_iterations X batch_size
# N_iterations = (N_epochs * train_data.shape[0])/batch_size
#N_iterations = (N_epochs * train_data.shape[0])/64 = 125000 for 1 epoch
parser.add_argument('--n_iterations', type=int, help='The number of iterations for training, the default is', required=False,default=50)
#default=5000000 )
parser.add_argument('--n_layers', type=int, help='The number of layers in your NN, the default is 5', required=False,default=6)
parser.add_argument('--n_hidden', type=int, help='The number of hidden layers in your NN, the default is 5', required=False,default=6)
parser.add_argument('--starting_learning_rate', type=float, help='Starting learning rate, the defulat is 10^-3', required=False,default=1.e-2)
parser.add_argument('--show_loss_plots', type=bool, help='Boolean to show the loss plots, default is False', required=False,default=False)
parser.add_argument('--save_model', type=bool, help='Boolean to save the trained model dictionary', required=False,default=False)
parser.add_argument('--save_loss_plots', type=bool, help='Boolean to save the loss plots', required=False,default=False)

args = parser.parse_args()


N = args.N
n_iterations = args.n_iterations
n_layers = args.n_layers
n_hidden = args.n_hidden
starting_learning_rate=args.starting_learning_rate
show_loss_plots=args.show_loss_plots
save_model=args.save_model
save_loss_plots=args.save_loss_plots



X       = ['genDatapT', 'genDataeta', 'genDataphi', 'genDatam', 'tau']

FIELDS  = {'RecoDatam' : {'inputs': X, 
                           'xlabel': r'$p_T$ (GeV)', 
                           'xmin': 0, 
                           'xmax':80},
           
           'RecoDatapT': {'inputs': ['RecoDatam']+X, 
                           'xlabel': r'$\eta$', 
                           'xmin'  : -8, 
                           'xmax'  :  8},
           
           'RecoDataeta': {'inputs': ['RecoDatam','RecoDatapT'] + X, 
                           'xlabel': r'$\phi$',
                           'xmin'  : -4,
                           'xmax'  :  4},
           
           'RecoDataphi': {'inputs': ['RecoDatam','RecoDatapT', 'RecoDataeta']+X,
                           'xlabel': r'$m$ (GeV)',
                           'xmin'  : 0, 
                           'xmax'  :20},
          }

target = 'RecoDataphi'
source  = FIELDS[target]
features= source['inputs']
print('FEATURES', features)
########
print('USING NEW DATASET')
train_data=pd.read_csv(DATA_DIR+'/train_data_10M_2.csv')
print('TRAINING FEATURES\n', train_data[features].head())
test_data=pd.read_csv(DATA_DIR+'/test_data_10M_2.csv')
valid_data=pd.read_csv(DATA_DIR+'/validation_data_10M_2.csv')

print('train set shape:',  train_data.shape)
print('validation set shape:', valid_data.shape)
print('test set shape:  ', test_data.shape)

# create a scaler for target
scaler_t = StandardScaler()
scaler_t.fit(train_data[target].to_numpy().reshape(-1, 1))

# create a scaler for inputs
scaler_x = StandardScaler()
scaler_x.fit(train_data[features])

# NB: undo scaling of tau, which is always the last feature
#this is a nice trick!
scaler_x.mean_[-1] = 0
scaler_x.scale_[-1]= 1

scalers = [scaler_t, scaler_x]

train_t, train_x =utils. split_t_x(train_data, target, features, scalers)
valid_t, valid_x = utils.split_t_x(valid_data, target, features, scalers)
test_t,  test_x  = utils.split_t_x(test_data,  target, features, scalers)

print('TARGETS ARE', train_t)
print()


print(train_t.shape, train_x.shape)
##########



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

# A simple wrapper around a model to make using the latter more
# convenient
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

y_label_dict ={'RecoDatapT':'$p(p_T)$'+' [ GeV'+'$^{-1} $'+']',
                    'RecoDataeta':'$p(\eta)$', 'RecoDataphi':'$p(\phi)$',
                    'RecoDatam':'$p(m)$'+' [ GeV'+'$^{-1} $'+']'}

loss_y_label_dict ={'RecoDatapT':'$p_T^{reco}$',
                    'RecoDataeta':'$\eta^{reco}$', 'RecoDataphi':'$\phi^{reco}$',
                    'RecoDatam':'$m^{reco}$'}


def plot_average_loss(traces, ftsize=18,save_loss_plots=save_loss_plots):
    
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
        plt.savefig('images/loss_curves/IQN_'+N+T+'_Consecutive_2.png')
        print('\nloss curve saved in images/loss_curves/IQN_'+N+target+'_Consecutive.png')
    # if show_loss_plots:
    #     plt.show()



def run(model, scalers, target, 
        train_x, train_t, 
        valid_x, valid_t, traces,
        n_batch=64, 
        n_iterations=n_iterations, 
        traces_step=500, 
        traces_window=500,
        save_model=save_model):

    learning_rate= starting_learning_rate
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate) 
    
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
    
    learning_rate=learning_rate/10
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate) 
    
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

    plot_average_loss(traces)

    if save_model:
        torch.save(model.state_dict(), 'trained_models/iqn_model_CONSECUTIVE10M_2_%s.dict' % target)
        print('\ntrained model dictionary saved in trained_models/iqn_model_CONSECUTIVE10M_2_%s.dict' % target)
    return utils.ModelHandler(model, scalers) 


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




def plot_model(dnn, target, src,
               fgsize=(6, 6), 
               ftsize=20,save_image=False, save_pred=False):

    eval_data=pd.read_csv('AUTOREGRESSIVE_m_Prime_pT_Prime_eta_Prime.csv')
    old_cols =['RecoDatam', 'RecoDatapT','RecoDataeta'] + X
    # eval_data=eval_data.reindex(old_cols)
    eval_data=eval_data[old_cols]
    print('EVALUATION DATA OLD INDEX\n', eval_data.head())

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

    # ax.hist(train_data['RecoDataphi'], 
    #         bins=xbins, 
    #         range=(xmin, xmax), 
    #         alpha=0.3, 
    #         color='blue', 
    #         density=True, 
    #         label='simulation')
   
    y = dnn(eval_data)
    eval_data['RecoDataphi']=y
    new_cols=['RecoDatam', 'RecoDatapT','RecoDataeta', 'RecoDataphi'] + X
    eval_data=eval_data.reindex(columns=new_cols)
    print('EVALUATION DATA NEW INDEX\n', eval_data.head())

    eval_data.to_csv('AUTOREGRESSIVE_m_Prime_pT_Prime_eta_Prime_phi_Prime.csv')
    eval_data.to_csv('FINAL_PREDICTIONS/    AUTOREGRESSIVE_m_Prime_pT_Prime_eta_Prime_phi_Prime.csv')

    if save_pred:
        pred_df = pd.DataFrame({T+'_predicted':y})
        pred_df.to_csv('predicted_data/dataset2/'+T+'_predicted_MLP_iter_5000000.csv')
    # ax.hist(y, 
    #         bins=xbins, 
    #         # range=(xmin, xmax), 
    #         alpha=0.3, 
    #         color='red', 
    #         density=True, 
    #         label='$y^\prime$')
    # ax.grid()
    # ax.legend()
    
    # plt.tight_layout()
    # if save_image:
    #     plt.savefig('images/'+T+'IQN_Consecutive_'+N+'.png')
    #     print('images/'+T+'IQN_Consecutive_'+N+'.png')
    # plt.show()
##########




def main():
    start=time.time()
    print('estimating phi\n')
    model =  utils.RegressionModel(
            nfeatures=train_x.shape[1], 
               ntargets=1,
               nlayers=n_layers, 
               hidden_size=n_hidden
               )

    traces = ([], [], [], [])

	# dnn = utils.ModelHandler(model, scalers)
    
    dnn = run(model, scalers, target, 
          train_x, train_t, 
          valid_x, valid_t, traces)

    plot_model( dnn, target, source)
    with open('phi_params.json','w') as f:
        f.write(json.dumps({'n_layers':n_layers, 'n_hidden':n_hidden, 'starting_learning_rate': starting_learning_rate,'n_iterations':n_iterations }))
    end=time.time()
    difference=end-start
    print('evaluating phi took ',difference/3600, 'hours')

if __name__ == "__main__":
    main()
