#!/bin/bash

#DAVIDSON
export DATA_DIR='/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/data'
#LOCAL
# export DATA_DIR='/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data'



python Train_Evaluate_m.py --n_iterations 500000 --n_layers 2 --n_hidden 4 --starting_learning_rate 2.1205283263244004e-03
python Train_Evaluate_pT.py --n_iterations 500000 --n_layers 2 --n_hidden 4 --starting_learning_rate 1.0404334991989074e-03
python Train_Evaluate_Eta.py --n_iterations 1
python Train_Evaluate_phi.py --n_iterations 1
