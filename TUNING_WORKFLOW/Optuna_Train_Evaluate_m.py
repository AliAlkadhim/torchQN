
import utils
import os, sys
import time
import os
import json
from numba import njit
start=time.time()
# the standard module for tabular data
import pandas as pd

# the standard module for array manipulation
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt
import optuna



# pytorch
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split
# linearly transform a feature to zero mean and unit variance
from sklearn.preprocessing import StandardScaler

# to reload modules
import importlib
# import mplhep as hep
# hep.style.use("CMS") # string aliases work too
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
           
           'RecoDataphi'  : {'inputs': ['RecoDatam', 'RecodatapT', 'RecoDataeta']+X,
                           'xlabel': r'$m$ (GeV)',
                           'xmin'  : 0, 
                           'xmax'  :20}
          }




target = 'RecoDatam'
source  = FIELDS[target]
features= source['inputs']

if N=='10M':
    train_data=pd.read_csv(DATA_DIR+'/train_data_10M.csv')
    test_data=pd.read_csv(DATA_DIR+'/test_data_10M.csv')
    valid_data=pd.read_csv(DATA_DIR+'/validation_data_10M.csv')

    print('train set shape:        %6d' % train_data.shape)
    print('validation set shape:   %6d' % valid_data.shape)
    print('test set shape:         %6d' % test_data.shape)

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
    print('TRAINING FEATURES', train_x)
    # valid_t, valid_x = utils.split_t_x(valid_data, target, features, scalers)
    test_t,  test_x  = utils.split_t_x(test_data,  target, features, scalers)

    print('TARGETS ARE', train_t)
    print()
    print('TRAINING FEATURES', train_x)

    print(train_t.shape, train_x.shape)

elif N=='10M_2':
    print('USING NEW DATASET')
    SUBSAMPLE=int(1e6)
    train_data=pd.read_csv(DATA_DIR+'/train_data_10M_2.csv', nrows=SUBSAMPLE)
    print('TRAINING FEATURES\n', train_data[features].head())
    test_data=pd.read_csv(DATA_DIR+'/test_data_10M_2.csv',nrows=SUBSAMPLE)
    # valid_data=pd.read_csv(DATA_DIR+'/validation_data_10M_2.csv',nrows=100)

    print('train set shape:',  train_data.shape)
    # print('validation set shape:', valid_data.shape)
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
    # valid_t, valid_x = utils.split_t_x(valid_data, target, features, scalers)
    test_t,  test_x  = utils.split_t_x(test_data,  target, features, scalers)

    print('TARGETS ARE', train_t)
    print()

    print(train_t.shape, train_x.shape)


else:
    print('You are using the 100,000 sample data!\n')
    data    = pd.read_csv(DATA_DIR+'/data_100k.csv')
    print('number of entries:', len(data))


    # Fraction of the data assigned as test data
    fraction = 20/100
    # Split data into a part for training and a part for testing
    train_data, test_data = train_test_split(data, 
                                            test_size=fraction)

    # Split the training data into a part for training (fitting) and
    # a part for validating the training.
    fraction = 5/80
    train_data, valid_data = train_test_split(train_data, 
                                            test_size=fraction)

    # reset the indices in the dataframes and drop the old ones
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)

    print('train set shape :   ',  train_data.shape)
    print('validation set shape:  ',  valid_data.shape)
    print('test set shape:         ' , test_data.shape)

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
    print('TARGETS SHAPE', train_t.shape)
    print()
    print('TRAINING FEATURES', train_x)
    print('TRAINING FEATURES SHAPE', train_x.shape)



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
    #tau=torch.rand(f.shape)
    return torch.mean(torch.where(t >= f, 
                                  tau * (t - f), 
                                  (1 - tau)*(f - t)))

# # function to validate model during training.
# def validate(model, avloss, inputs, targets):
#     # make sure we set evaluation mode so that any training specific
#     # operations are disabled.
#     model.eval() # evaluation mode
    
#     with torch.no_grad(): # no need to compute gradients wrt. x and t
#         x = torch.from_numpy(inputs).float()
#         t = torch.from_numpy(targets).float()
#         # remember to reshape!
#         o = model(x).reshape(t.shape)
#     return avloss(o, t, x)

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

class Engine:
    """loss, training and evaluation"""
    def __init__(self, model, optimizer, batch_size):
                 #, device):
        self.model = model
        #self.device= device
        self.optimizer = optimizer
        self.batch_size=batch_size
        
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
        """the training function: takes the training dataloader"""
        self.model.train()
        final_loss = 0
        for iteration in range(n_iterations):
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
        """the training function: takes the training dataloader"""
        self.model.eval()
        final_loss = 0
        for iteration in range(n_iterations):
            batch_x, batch_t = get_batch(x, t,  self.batch_size)
            inputs=torch.from_numpy(batch_x).float()
            targets=torch.from_numpy(batch_t).float()
            outputs = self.model(inputs)
            loss =average_quantile_loss(outputs, targets, inputs)
            final_loss += loss.item()
        return final_loss / self.batch_size



EPOCHS=1
def run_train(params, save_model=False):
    """For tuning the parameters"""

    model =  utils.RegularizedRegressionModel(
              nfeatures=train_x.shape[1], 
                ntargets=1,
                nlayers=params["nlayers"], 
                hidden_size=params["hidden_size"],
                dropout=params["dropout"]
                )
    learning_rate= params["learning_rate"]

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"]) 
    eng=Engine(model, optimizer, batch_size=params["batch_size"])
    best_loss = np.inf
    early_stopping_iter=10
    early_stopping_coutner=0

    for epoch in range(EPOCHS):
      train_loss = eng.train(train_x, train_t)
      valid_loss=eng.evaluate(test_x, test_t)
      print(f"{epoch} \t {train_loss} \t {valid_loss}")
      if valid_loss<best_loss:
        best_loss=valid_loss
        if save_model:
          model.save(model.state_dict(), "model_m.bin")
      else:
        early_stopping_coutner+=1
      if early_stopping_coutner > early_stopping_iter:
        break
    return best_loss

# run_train()

def objective(trial):
  params = {
      "nlayers": trial.suggest_int("nlayers",1,24),      
      "hidden_size": trial.suggest_int("hidden_size", 2, 3000),
      "dropout": trial.suggest_float("dropout", 0.1,0.5),
      "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
      "batch_size": trial.suggest_int("batch_size", 32, 4000)
      
  }
  # all_losses=[]

  temp_loss = run_train(params,save_model=False)
    # all_losses.append(temp_loss)
  return temp_loss




y_label_dict ={'RecoDatapT':'$p(p_T)$'+' [ GeV'+'$^{-1} $'+']',
                    'RecoDataeta':'$p(\eta)$', 'RecoDataphi':'$p(\phi)$',
                    'RecoDatam':'$p(m)$'+' [ GeV'+'$^{-1} $'+']'}

loss_y_label_dict ={'RecoDatapT':'$p_T^{reco}$',
                    'RecoDataeta':'$\eta^{reco}$', 'RecoDataphi':'$\phi^{reco}$',
                    'RecoDatam':'$m^{reco}$'}



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


def run_train_best_model(best_params, save_model=False, EPOCHS=2):
    """For tuning the parameters"""

    model =  utils.RegressionModel(
              nfeatures=train_x.shape[1], 
                ntargets=1,
                nlayers=best_params["nlayers"], 
                hidden_size=best_params["hidden_size"])
    learning_rate= best_params["learning_rate"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    eng=Engine(model, optimizer, batch_size=64)
    best_loss = np.inf
    early_stopping_iter=10
    early_stopping_coutner=0

    for epoch in range(EPOCHS):
      train_loss = eng.train(train_x, train_t)
      valid_loss=eng.evaluate(test_x, test_t)
      print(f"{epoch} \t {train_loss} \t {valid_loss}")
      if valid_loss<best_loss:
        best_loss=valid_loss
        if save_model:
          model.save(model.state_dict(), "model_m.bin")
      else:
        early_stopping_coutner+=1
      if early_stopping_coutner > early_stopping_iter:
        break
    return utils.ModelHandler(model, scalers)



def evaluate_model(dnn):
    #for pt, the evaluation data is just the test data at the approriate features
    eval_data=pd.read_csv('AUTOREGRESSIVE_pT_Prime_eta_Prime_phi_Prime.csv')
    
    old_cols =['RecoDatapT', 'RecoDataeta', 'RecoDataphi','genDatapT','genDataeta','genDataphi','genDatam','tau']
    # eval_data=eval_data.reindex(old_cols)
    eval_data=eval_data[old_cols]
    print('EVALUATION DATA OLD INDEX\n', eval_data.head())

    y = dnn(eval_data)
    #evaluate before adding the new column

    y = dnn(eval_data)
    eval_data['RecoDatam']=y
    new_cols=['RecoDatapT','RecoDataeta','RecoDataphi', 'RecoDatam','genDatapT', 'genDataeta', 'genDataphi', 'genDatam', 'tau']
    eval_data=eval_data.reindex(columns=new_cols)
    print('EVALUATION DATA NEW INDEX\n', eval_data.head())
    eval_data.to_csv('AUTOREGRESSIVE_pT_Prime_eta_Prime_phi_Prime_m_Prime.csv')
    print('saved AUTOREGRESSIVE_pT_Prime_eta_Prime_phi_Prime_m_Prime.csv')


def main():
    print('Getting best hyperparameters for m\n')
    study=optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3000)
    best_trial = study.best_trial
    print('best model parameters', best_trial.params)

    best_params=best_trial.params#this is a dictionary
    filename='best_params/m_best_params_3kTrials.csv'
    param_df=pd.DataFrame({
        'n_layers':best_params["nlayers"], 
                            'hidden_size':best_params["hidden_size"], 
                            'dropout':best_params["dropout"],
                            'learning_rate': best_params["learning_rate"], 
                            'batch_size':best_params["batch_size"] },
                                    index=[0]
    )

    param_df.to_csv(filename)   
    # best_param_mass_df=pd.DataFrame(best_params).to_csv('best_params_mass.csv')
    # best_model = utils.RegressionModel(nfeatures=train_x.shape[1], 
    #                 ntargets=1,
    #                 nlayers=best_params["nlayers"], 
    #                 hidden_size=best_params["hidden_size"])

    # dnn = run_train_best_model(best_params, save_model=False, EPOCHS=2)
    # evaluate_model(dnn)


    

    end=time.time()
    difference=end-start
    print('Tuning m took ',difference, 'seconds')
if __name__ == "__main__":
    main()
