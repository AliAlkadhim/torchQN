import pandas as pd
import numpy as np
import os
import torch
# import data_loader_two_by_two as dat
import data_loader_IQN as dat_IQN
# import framework.config as config
import matplotlib.pyplot as plt
from numba import njit

import utils


# # import framework.framework as framework
# import framework.regressionframework as framework
# #import the directory_framework.filename_framework (becauase the directory is viewed as a package because it has __init__.py)
# import framework.layer as layer
# import framework.activation as activation
# import framework.loss_funcs as loss_funcs

################################# Open the full dataset to find the range of each feature ########
X       = ['genDatapT', 'genDataeta', 'genDataphi', 'genDatam', 'tau']

FIELDS  = {'RecoDatam' : {'inputs': X, 
                        'xlabel': r'$p_T$ (GeV)', 
                        'xmin': 0, 
                        'xmax':80},
        
        'RecoDatapT': {'inputs': ['RecoDatam']+X, 
                        'xlabel': r'$\eta$', 
                        'xmin'  : 0, 
                        'xmax'  :  120},
        
        'RecoDataeta': {'inputs': ['RecoDatam','RecoDatapT'] + X, 
                        'xlabel': r'$\phi$',
                        'xmin'  : -4,
                        'xmax'  :  4},
        
        'RecoDataphi'  : {'inputs': ['RecoDatam', 'RecodatapT', 'RecoDataeta']+X,
                        'xlabel': r'$m$ (GeV)',
                        'xmin'  : 0, 
                        'xmax'  :20}
        }

# DATA_DIR = os.path.join(os.getcwd(),'data')
# DATA_DIR='data/'
# os.environ["DATA_DIR"]="/home/ali/Desktop/Pulled_Github_Repositorie/IQN_HEP/Davidson/data"
# DATA_DIR=os.environ["DATA_DIR"]
DATA_DIR="/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data/"
target = 'RecoDatam'
source  = FIELDS[target]
features= source['inputs']
########
print('USING NEW DATASET')
train_data=pd.read_csv(DATA_DIR + '/train_data_10M_2.csv' ,
                                #  nrows=1000,
                       usecols=FIELDS[target]['inputs']
                       )
print('TRAINING FEATURES\n', train_data[features].head())
print('train set shape:',  train_data.shape)

# EXPECTED_VALUES_RANGE =(np.min(np.min( train_data[features])), np.max(np.max( train_data[features])) )
# print('EXPECTED_VALUES_RANGE \n' , EXPECTED_VALUES_RANGE)



@njit
def normalize_IQN(values, expected_input_range):
    expected_range=expected_input_range
    expected_min, expected_max = expected_range
    scale_factor = expected_max - expected_min
    offset = expected_min
    scaled_values = (values - offset)/scale_factor 
    return scaled_values
    
    
def denormalize_IQN(normalized_values, expected_input_range):
    expected_range=expected_input_range
    expected_min, expected_max = expected_range
    scale_factor = expected_max - expected_min
    offset = expected_min
    return normalized_values  * scale_factor + offset

def normalize_IQN_DF(DF):
    
    for i in range(DF.shape[1]):
        col = DF.iloc[:,i]
        expectd_col_min = np.min(col)
        expected_col_max = np.max(col)
        normed_col = normalize_IQN(col, expected_input_range=(expectd_col_min, expected_col_max) )
        DF.iloc[:,i] = normed_col
    return DF

def average_quantile_loss(f, t, x):
    # f and t must be of the same shape
    tau = x.T[-1] # last column is tau.
    return torch.mean(torch.where(t >= f, 
                                  tau * (t - f), 
                                  (1 - tau)*(f - t)))
    
    

training_set_features, training_set_targets, evaluation_set_features, evaluation_set_targets = dat_IQN.get_data_set(target)

# train_generator, eval_generator = dat_IQN.get_data_set()

sample_x=next(training_set_features())#this is just to get the dimenstions of one batch
sample_y=next(training_set_targets())
#(batchsize,5) for mass
print('sample x shape', sample_x.shape)
print('sample x shape', sample_y.shape)

n_features = sample_x.shape[1]


def train(model, optimizer, niterations):
    for n_iter in range(niterations):
        model.train()
        x_batch = next(training_set_features())        
        # x_batch= normalize_IQN(x_batch, expected_input_range=EXPECTED_VALUES_RANGE)
        y_batch = next(training_set_targets())
        # y_batch = normalize_IQN(y_batch, expected_input_range=EXPECTED_VALUES_RANGE)
        
        with torch.no_grad():
            x = torch.from_numpy(x_batch).float()
            y = torch.from_numpy(y_batch).float()
        #from below has to be inside     
        outputs = model(x)
        outputs=outputs.reshape(y.shape)
        
        cost = average_quantile_loss(outputs, y, x)
        optimizer.zero_grad()
        cost.backward()
        
        print(cost, end='\n')
            
            


n_layers=1
n_hidden=64
niterations=int(1e2)


model =  utils.RegularizedRegressionModel(
    nfeatures=n_features, 
    ntargets=1,
    nlayers=n_layers, 
    hidden_size=n_hidden)

learning_rate=int(1e-2)
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate) 

train(model, optimizer, niterations)




def plot_model(dnn, target, src,
               fgsize=(6, 6), 
               ftsize=20,save_image=False, 
               save_pred=False,
               show_plot=True):
    eval_data_orig=pd.read_csv(DATA_DIR+'/test_data_10M_2.csv')
    expected_ymin = np.min(eval_data_orig['RecoDatam'])
    expected_ymax = np.max(eval_data_orig['RecoDatam'])
    EXPECTED_VALUES_RANGE = (expected_ymin, expected_ymax)
    
    ev_features=X
    #['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','tau']
    eval_data_df=eval_data_orig[ev_features]

    #normalize the evaluation data
    eval_data_df = normalize_IQN_DF(eval_data_df)
    
    eval_data=np.array(eval_data_df)
    # eval_data_normalized = normalize_IQN(eval_data, expected_input_range=EXPECTED_VALUES_RANGE)
    eval_data_normalized = eval_data
    eval_data_normalized = torch.Tensor(eval_data_normalized)
    
    
    print('EVALUATION DATA OLD INDEX\n', eval_data_df.head())

   
    dnn.eval()
        
    y = dnn(eval_data_normalized).view(-1, ).detach().numpy()
    
    y_denormalized = denormalize_IQN(y, EXPECTED_VALUES_RANGE)
    
    eval_data_df['RecoDatam']=y_denormalized
    new_cols= ['RecoDatam'] + X
    eval_data_df=eval_data_df.reindex(columns=new_cols)
    print('EVALUATION DATA NEW INDEX\n', eval_data_df.head())

    eval_data_df.to_csv('AUTOREGRESSIVE_m_Prime.csv')


    # if save_pred:
        # pred_df = pd.DataFrame({T+'_predicted':y})
        # pred_df.to_csv('predicted_data/dataset2/'+T+'_predicted_MLP_iter_5000000.csv')
    
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
        ax.set_xlabel('reco jet mass', fontsize=ftsize)
        # ax.set_ylabel(y_label_dict[target], fontsize=ftsize)
        ax.hist(eval_data_orig['RecoDatam'], 
            bins=xbins, 
            range=(xmin, xmax), 
            alpha=0.3, 
            color='blue', 
            density=True, 
            label='truth')
        ax.hist(y_denormalized, 
                bins=xbins, 
                range=(xmin, xmax), 
                alpha=0.3, 
                color='red', 
                density=True, 
                label='$IQN prediction$')
        ax.grid()
        ax.legend()
        
        plt.tight_layout()
    
    # if save_image:
    #     plt.savefig('images/'+T+'IQN_Consecutive_'+N+'.png')
    #     print('images/'+T+'IQN_Consecutive_'+N+'.png')
    
    if show_plot:
        plt.show()
    

plot_model( model, target, source)
