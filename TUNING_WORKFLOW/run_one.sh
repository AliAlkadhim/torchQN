#!/bin/bash

#DAVIDSON
#export DATA_DIR='/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/data'
#LOCAL
export DATA_DIR='/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data'

conda init bash
source ~/.bashrc
conda activate torch_env
conda info --envs


#python Optuna_Train_Evaluate_m.py 
python Train_Evaluate_m.py --n_iterations 1000
# python plot_results_m.py --T RecoDatam

######################EXPERIMENTAL
# python OptunaTunepT.py
