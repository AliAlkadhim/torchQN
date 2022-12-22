
NFEATURES=train_x.shape[1]

def load_untrained_model():
    model=TrainingRegularizedRegressionModel(nfeatures=NFEATURES, ntargets=1,
                               nlayers=n_layers, hidden_size=hidden_size, dropout=dropout)
    print(model)
    return model

model=load_untrained_model()



# optimizer_name =  'Adam'
best_learning_rate =  float(BEST_PARAMS["learning_rate"])
momentum=float(BEST_PARAMS["momentum"]) 
best_optimizer_temp = getattr(torch.optim, optimizer_name)(model.parameters(), lr=best_learning_rate,momentum=momentum)
batch_size = int(BEST_PARAMS["batch_size"])


# BATCHSIZE=10000
BATCHSIZE=batch_size 
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
    print("%10s\t%10s\t%10s" %           ('iteration', 'train-set', 'test-set'))
    
    for ii in range(n_iterations):

        # set mode to training so that training specific 
        # operations such as dropout are enabled.
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
            
    print()      
    return (xx, yy_t, yy_v, yy_v_avg)


@time_type_of_func(tuning_or_training='training')
def run(model, 
        train_x, train_t, 
        valid_x, valid_t, traces,
        n_batch=BATCHSIZE, 
        n_iterations=n_iterations, 
        traces_step=200, 
        traces_window=200,
        save_model=False):

    learning_rate= best_learning_rate
    #add weight decay (important regularization to reduce overfitting)
    L2=1e-3
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=best_learning_rate, momentum=momentum, weight_decay=L2)
    
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
    
    learning_rate=learning_rate/100
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=best_learning_rate, momentum=momentum)
    #10^-4
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


#     learning_rate=learning_rate/100
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
#     #10^-6
#     traces = train(model, optimizer, 
#                       average_quantile_loss,
#                       get_batch,
#                       train_x, train_t, 
#                       valid_x, valid_t,
#                       n_batch, 
#                   n_iterations,
#                   traces,
#                   step=traces_step, 
#                   window=traces_window)

    # plot_average_loss(traces)

    if save_model:
        filename='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
        PATH = os.path.join(IQN_BASE, 'trained_models', filename)
        torch.save(model.state_dict(), PATH)
        print('\ntrained model dictionary saved in %s' % PATH)
    return  model


# ## See if trainig works on T ratio





IQN_trace=([], [], [], [])
traces_step = 800
traces_window=traces_step
n_iterations=100000
IQN = run(model=model,train_x=train_x, train_t=train_t_ratio, 
        valid_x=test_x, valid_t=test_t_ratio, traces=IQN_trace, n_batch=BATCHSIZE, 
        n_iterations=n_iterations, traces_step=traces_step, traces_window=traces_window,
        save_model=False)


# ## Save trained model (if its good, and if you haven't saved above) and load trained model (if you saved it)


filename='Trained_IQNx4_%s_%sK_iter.dict' % (target, str(int(n_iterations/1000)) )
trained_models_dir='trained_models'
mkdir(trained_models_dir)
PATH = os.path.join(IQN_BASE,trained_models_dir , filename)

@debug
def save_model(model):
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
    model =  TrainingRegularizedRegressionModel(
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



save_model(IQN)

####################### Evaluation below ####################### 



# plt.hist(valid_t_ratio, label='post-z ratio target');
# for i in range(NFEATURES):
#     plt.hist(valid_x[:,i], label =f"feature {i}", alpha=0.35)
# plt.legend();plt.show()



def simple_eval(model):
    model.eval()
    valid_x_tensor=torch.from_numpy(valid_x).float()
    pred = IQN(valid_x_tensor)
    p = pred.detach().numpy()
    fig, ax = plt.subplots(1,1)
    label=FIELDS[target]['ylabel']
    ax.hist(p, label=f'Predicted post-z ratio for {label}', alpha=0.4, density=True)
    orig_ratio = z(T('m', scaled_df=train_data_m))
    print(orig_ratio[:5])
    ax.hist(orig_ratio, label = f'original post-z ratio for {label}', alpha=0.4,density=True)
    ax.grid()
    set_axes(ax, xlabel='predicted $T$')
    print('predicted ratio shape: ', p.shape)
    return p
    
p = simple_eval(IQN)
plt.show()


# In[194]:


# IQN.eval()
# valid_x_tensor=torch.from_numpy(valid_x).float()
# pred = IQN(valid_x_tensor)
# p = pred.detach().numpy()
# plt.hist(p, label='predicted $T$ ratio');plt.legend();plt.show()


# $$
#         f_{\text{IQN}} (\mathcal{O}) =  z \left( \frac{\mathbb{L} (\mathcal{O}^{\text{reco}}) +10 }{\mathbb{L}(\mathcal{O}^{\text{gen}}) +10} \right),
# $$
# 
# 
# So, to de-scale, (for our observable $\mathcal{O}=m$ ),
# 
# $$
#     m^{\text{predicted}} = \mathbb{L}^{-1} \left[ z^{-1} (f_{\text{IQN}} ) \left[ \mathbb{L} (m^\text{gen})+10 \right] -10 \right]
# $$
# 
# Note that $z^{-1} (f_{\text{IQN}} )$ should use the mean and std of the ratio thing for the target 
# 
# $$z^{-1} (f_{\text{IQN}} ) = z^{-1}\left( y_{pred}, \text{mean}=\text{mean}(\mathbb{T}(\text{target_variable})), std=std (\mathbb{T}(\text{target_variable} ) \right)$$

# In[195]:


def z_inverse(xprime, mean, std):
    return xprime * std + mean


# In[196]:






raw_train_data=pd.read_csv(os.path.join(DATA_DIR,'train_data_10M_2.csv'),
                      usecols=all_cols,
                      nrows=SUBSAMPLE
                      )

raw_test_data=pd.read_csv(os.path.join(DATA_DIR,'test_data_10M_2.csv'),
                      usecols=all_cols,
                     nrows=SUBSAMPLE
                     )
raw_test_data.describe()
m_reco = raw_test_data['RecoDatam']
m_gen = raw_test_data['genDatam']
plt.hist(m_reco,label=r'$m_{gen}^{test \ data}$');plt.legend();plt.show()


# 
# 
# Apply the descaling formula for our observable
# 
# $$
#     m^{\text{predicted}} = \mathbb{L}^{-1} \left[ z^{-1} (f_{\text{IQN}} ) \left[ \mathbb{L} (m^\text{gen})+10 \right] -10 \right]
# $$
# 
# * First, calculate $z^{-1} (f_{\text{IQN}} )$

# In[198]:


print(valid_t_ratio.shape, valid_t_ratio[:5])


# In[199]:


orig_ratio = T('m', scaled_df=train_data_m)
orig_ratio[:5]


# In[200]:


z_inv_f =z_inverse(xprime=p, mean=np.mean(orig_ratio), std=np.std(orig_ratio))
z_inv_f[:5]


# * Now 
# 
# $$\mathbb{L}(\mathcal{O^{\text{gen}}}) = \mathbb{L} (m^{\text{gen}})$$
# 

# In[201]:


L_obs = L(orig_observable=m_gen, label='m')
L_obs[:5]


# In[202]:


print(L_obs.shape, z_inv_f.shape)


# In[203]:


z_inv_f = z_inv_f.flatten();print(z_inv_f.shape)


# * "factor" $ = z^{-1} (f_{\text{IQN}} ) \left[ \mathbb{L} (m^\text{gen})+10 \right] -10 $

# In[204]:


factor = (z_inv_f * (L_obs  + 10) )-10
factor[:5]


# In[205]:


m_pred = L_inverse(L_observable=factor, label='m')
# pT_pred=get_finite(pT_pred)


# In[206]:


m_pred


# In[207]:


plt.hist(m_pred.flatten(),label='predicted',alpha=0.3);
plt.hist(m_reco,label=r'$m_{reco}^{test \ data}$',alpha=0.3);

plt.legend();plt.show()


# ------------------
# ### Paper plotting

# In[208]:


range_=[0,25]
bins=50
data=raw_train_data
YLIM=(0.8,1.2)
data = data[['RecoDatapT','RecoDataeta','RecoDataphi','RecoDatam']]
data.columns = ['realpT','realeta','realphi','realm']
REAL_DIST=data['realm']
norm_data=data.shape[0]
AUTOREGRESSIVE_DIST = m_pred
norm_IQN=AUTOREGRESSIVE_DIST.shape[0]
norm_autoregressive=AUTOREGRESSIVE_DIST.shape[0]
norm_IQN=norm_autoregressive
print('norm_data',norm_data,'\nnorm IQN',norm_IQN,'\nnorm_autoregressive', norm_autoregressive)


# In[209]:


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


# In[210]:


real_label_counts_m, predicted_label_counts_m, label_edges_m = get_hist_simple('m')


# In[211]:


def plot_one_m():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
    ax1.step(label_edges_m, real_label_counts_m/norm_data, where="mid", color="k", linewidth=0.5)# step real_count_pt
    ax1.step(label_edges_m, predicted_label_counts_m/norm_IQN, where="mid", color="#D7301F", linewidth=0.5)# step predicted_count_pt
    ax1.scatter(label_edges_m, real_label_counts_m/norm_data, label="reco",  color="k",facecolors='none', marker="o", s=5, linewidth=0.5)
    ax1.scatter(label_edges_m,predicted_label_counts_m/norm_IQN, label="predicted sbatch 1", color="#D7301F", marker="x", s=5, linewidth=0.5)
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
    ax2.set_ylabel(r"$\frac{\textnormal{predicted}}{\textnormal{reco}}$")
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


# In[212]:


plot_one_m()


# -------------
# -------------
# -------------
# 

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
