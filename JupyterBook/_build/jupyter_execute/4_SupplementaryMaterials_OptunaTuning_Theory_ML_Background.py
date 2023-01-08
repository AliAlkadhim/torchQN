#!/usr/bin/env python
# coding: utf-8

# # Chapter 4: Optional Supplementary Material

# In[ ]:





# # 4.1: Scaling
# 
# scaling (or standarization, normalization) is someimes done in the following way:
# $$ X' = \frac{X-X_{min}}{X_{max}-X_{min}} \qquad \rightarrow \qquad X= X' (X_{max}-X_{min}) + X_{min}$$

# In[26]:


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
# 2. Fit on the train set, and transform everything according to the train set, that is, get the mean and std, ( optionally and min and max or other quantities) of each feature (column) of each of the train set, and standarize everything according to that.
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

# In[27]:


use_svg_display()
show_jupyter_image('images/scaling_forNN.jpg', width=2000,height=500)


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

# # 4.2: Auto-Tuning with Optuna

# In[ ]:





# --------
# -------
# 
# ## Hyperparameter Training Workflow

# We should not touch our test set until we have chosen our hyperparameters (have done model selection) using the valid set (also, we can't tune the model using the train set since it obviously will just overfit the train set). Then in the training process we evaluate on test set to keep ourselves honest and to see if we are overfitting the train set. 
# 
# It seems that one of the big challenges in this project is generalization (or overfitting). ways to reduce overfitting and improve generalization (that is, make the generalization error $R[f]$ closer to the training error $R_{\text{emp}}$ is: 
# 
# 1. Reducing model complexity (e.g. using a linear model instead of a non-linear model, and user fewer model parameters). 2. using more training data. 3. feature selecting (i.e. using fewer features, since some of the features might be irrelevant or redundant). 4. Data augmentation, i.e. generating additional training examples through e.g. horizontal flipping/random cropping for images, or adding noise to training examples, all of which increase the size and diversity of the training set. 5. Regularization techniques, such as adding a penalty to the weights of the model during training (weight decay). This penalty could be e.g. L1 or L2 norm of the weights. 5. Cross valudation: dividing the training set into several smaller sets, training the model on one set, and evaluating it on the others. (can use e.g. `sklearn.model_selection.KFold` instead of doing it from scratch). 
# 
# Other than overfitting, the name of the game for training of course is to minimize the loss with respect to the weights. INPUT (features) is forward propagated (meaning each layer is basically a matrix of the size of the weights is a matrix of $( N_inputs + 1) X (N_outputs)$ (and the inputs) is $w[N_inputs]+b$ which is the rows, and the outputs ($y=\mathbf{w}+b$) will have shape $[N_outputs]$  through each layer into the OUTPUT (target). More explicitly the shape of output $y$ of each layer will be
# 
# $$[y] = [1 \times (N_{input}+1) ] \cdot [(N_{input} +1) \times N_{output}] = [1\times N_{output}]  $$
# 
# Where $N_{input}$ and $N_{output}$ are the numbers of input and output features, taken to be rows and columns of the matrices, respectively. The $+1$ in $ (N_{input}+1) $ is for the bias column.
# 
# 
# Forward Prop: pass.
# INPUT $\mathbf{x}$ -> .... -> a bunch of layers ... -> OUTPUT $y$
# 
# 
# Backprop.: Going backward from the output layer to the input layer. Basically, for one layer, $Backprop(dLoss/dy) = dLoss/dx$, where $x$ and $y$ are the inputs and outputs for one layer. This will use, in the simplest case, $dLoss/dx = dLoss/dy \ dy/dx$ where $ dy/dx$ will be in terms of the weights of the current layer. $dLoss/dw = dLoss/dy \ dy/dw$, and then the weights are updated as to minimize the loss, e.g. for SGD: $w \leftarrow w - (dLoss/dw \ \eta$ where $\eta$ is the learning rate.
# 
# the dLoss/dx becomes the dLoss/dy of the next layer:
# 
# X layer1 - > layer_2 .... -> layer_n -> y
# <-...  dL/dx  <- BP dL/dy  <- dL/dx   <-BP  dL/dy

# In[53]:


def get_tuning_sample():
    sample=int(200000)
    # train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample
    get_whole=True
    if get_whole:
        train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample = train_x, train_t_ratio, valid_x, valid_t_ratio
    else:
        train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample=train_x[:sample], train_t_ratio[:sample], valid_x[:sample], valid_t_ratio[:sample]
    return train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample

train_x_sample, train_t_ratio_sample, valid_x_sample, valid_t_ratio_sample = get_tuning_sample()
print(train_x_sample.shape)


# Need to use test set *and* validation set for tuning.
# 
# Note that hyperparameters are usually not directly transferrable across architectures and datasets. 

# In[55]:


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
        nlayers_max,n_hidden_max, batch_size_max=int(24),int(350), int(2e5)
    else:
        nlayers_max,n_hidden_max, batch_size_max=int(6),int(256), int(3e4)

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
    n_trials=100
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial
    print('best model parameters', best_trial.params)

    best_params=best_trial.params#this is a dictionary
    #save best hyperapameters in a pandas dataframe as a .csv
    if save_best_params:
        tuned_dir = os.path.join(IQN_BASE,'best_params')
        mkdir(tuned_dir)
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


# In[56]:


study= tune_hyperparameters(save_best_params=True)


# In[ ]:


optuna.visualization.plot_parallel_coordinate(study)


# --------------
# ------------
# -------
# 

# In[413]:


# BEST_PARAMS = {'nlayers': 6, 'hidden_size': 2, 'dropout': 0.40716885971031636, 'optimizer_name': 'Adam', 'learning_rate': 0.005215585403055171, 'batch_size': 1983}


# In[183]:


# def get_model_params_tuned()

tuned_dir = os.path.join(IQN_BASE,'best_params')
tuned_filename=os.path.join(tuned_dir,'best_params_mass.csv')
# BEST_PARAMS = pd.read_csv(os.path.join(IQN_BASE, 'best_params','best_params_Test_Trials.csv'))
BEST_PARAMS=pd.read_csv(tuned_filename)
print(BEST_PARAMS)

n_layers = int(BEST_PARAMS["n_layers"]) 
hidden_size = int(BEST_PARAMS["hidden_size"])
dropout = float(BEST_PARAMS["dropout"])

optimizer_name = BEST_PARAMS["optimizer_name"]
print(type(optimizer_name))
optimizer_name = BEST_PARAMS["optimizer_name"].to_string().split()[1]


# In[238]:


#make sure you have plotly installed with jupyterlab support. For jupyterlab3.x:
# conda install "jupyterlab>=3" "ipywidgets>=7.6"

# for JupyterLab 2.x renderer support:
# jupyter labextension install jupyterlab-plotly@5.11.0 @jupyter-widgets/jupyterlab-manager
#conda install -c plotly plotly=5.11.0
from optuna import visualization

visualization.plot_param_importances(study)

