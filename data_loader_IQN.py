import numpy as np
import pandas as pd
import os
# import framework.utils as utils

def get_batch(x, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]
    # batch_x.T[-1] = np.random.uniform(0, 1, batch_size)
    return batch_x

def split_t_x(df, target, source):
    # change from pandas dataframe format to a numpy array
    t = np.array(df[target].to_numpy().reshape(-1, 1))
    #where scaler_t is a StandardScaler() object, which has the .transorm method
    x = np.array(df[source])
    # t = t.reshape(-1,)
    return t, x

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



def get_data_set():
    # os.environ["DATA_DIR"]="/home/ali/Desktop/Pulled_Github_Repositorie/IQN_HEP/Davidson/data"
    # DATA_DIR=os.environ["DATA_DIR"]
    # DATA_DIR = 'data'
    DATA_DIR="/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data/"

    target = 'RecoDatam'
    source  = FIELDS[target]
    features= source['inputs']
    ########
    print('USING NEW DATASET')
    train_data=pd.read_csv(DATA_DIR + '/train_data_10M_2.csv' )
    print('TRAINING FEATURES\n', train_data[features].head() )

    test_data=pd.read_csv(DATA_DIR+'/test_data_10M_2.csv')
    valid_data=pd.read_csv(DATA_DIR+'/validation_data_10M_2.csv')

    print('train set shape:',  train_data.shape)
    print('validation set shape:', valid_data.shape)
    print('test set shape:  ', test_data.shape)    





    n_examples= int(8e6)
    batchsize=10
    N_batches = 10
    #N_batches=n_examples/batch_size
    train_t, train_x = split_t_x(train_data, target, features)
    
    test_t,  test_x  = split_t_x(test_data,  target, features)

    print('PRE-NORMALIZATION train_x\n', train_x)
    print(train_x.shape)

    
    #IF I WANT TO NORMALIZE EACH COLUMN INDEPENDENTLY DO THIS
    # for i in range(train_x.shape[1]):
    #     train_x[:,i] = utils.normalize_IQN(train_x[:,i] , expected_input_range = (np.min( train_x[:,i] ), np.max( train_x[:,i] ))
    #     )

    print('POST-NORMALIZATION train_x\n', train_x)
    print('new feature means', np.mean(train_x,axis=1))
    print('new feature maxes', np.max(train_x,axis=1))
    # train_t = utils.normalize_IQN(train_t, expected_input_range=(np.min(train_t), np.max(train_t)))
    print('new targets', train_t)


    #do same for test 
    # for i in range(test_x.shape[1]):
    #     test_x[:,i] = utils.normalize_IQN(test_x[:,i] , expected_input_range = (np.min( test_x[:,i] ), np.max( test_x[:,i] )))

    # test_t = utils.normalize_IQN(test_t, expected_input_range=(np.min(test_t),np.max(test_t)))


    def training_set_features():
        #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            #get a random batch of the defined size
            batch_x = get_batch(train_x, batchsize)
            #print('batch_x', batch_x)
            #index of one of the items in our examples
            yield batch_x

    def evaluation_set_features():
        #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            batch_x = get_batch(test_x,batchsize)
            #index of one of the items in our examples
            yield batch_x
            
        
    def training_set_targets():
            #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            #get a random batch of the defined size
            batch_x = get_batch(train_t, batchsize)
            #print('batch_x', batch_x)
            #index of one of the items in our examples
            yield batch_x
            
    def evaluation_set_targets():
            #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            #get a random batch of the defined size
            batch_x = get_batch(test_t, batchsize)
            #print('batch_x', batch_x)
            #index of one of the items in our examples
            yield batch_x

    return training_set_features, training_set_targets, evaluation_set_features, evaluation_set_targets







if __name__ == '__main__':
    training_set_features, training_set_targets, evaluation_set_features, evaluation_set_targets=get_data_set()
    sample_x=next(training_set_features)
    print('sample_x', training_set_features)
    print('sample_x shape', training_set_features.shape)
    print()
    sample_y=next(training_set_targets())
    print('sample_y', training_set_targets)
    # print('sample_y shape', training_set_targets.shape)
