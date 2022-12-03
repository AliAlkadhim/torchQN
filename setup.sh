#!/bin/bash
export IQN_BASE=/home/ali/Desktop/Pulled_Github_Repositories/torchQN
#DAVIDSON
#export DATA_DIR='/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/data'
#LOCAL
export DATA_DIR='/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data'
echo 'DATA DIR'
ls -l $DATA_DIR
#ln -s $DATA_DIR $IQN_BASE, if you want
#conda create env -n torch_env -f torch_env.yml
#conda activate torch_env
mkdir -p ${IQN_BASE}/images/loss_plots ${IQN_BASE}/trained_models  ${IQN_BASE}/hyperparameters ${IQN_BASE}/predicted_data
tree $IQN_BASE
