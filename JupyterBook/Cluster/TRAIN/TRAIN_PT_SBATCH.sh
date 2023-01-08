#!/bin/bash
# echo "shell" $0                                                                                                                                                                                         $
# rnd=$(($1 + 1)) 
# CHANGE JOB NAME
#SBATCH --job-name "DEC19_m"
#SBATCH --output "log_AUTOREGRESSIVE/output_sbatch_Autoregressive_.%j.log"
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 37
#SBATCH --mem-per-cpu 30g
#SBATCH --oversubscribe
#SBATCH --priority=TOP



conda init bash
source ~/.bashrc
conda activate torch_env
conda info --envs


#DAVIDSON
export DATA_DIR='/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/data'
#LOCAL
#export DATA_DIR='/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data'

srun python TrainpT.py
#python Optuna_Train_Evaluate_m.py

### DO scontrol -d show job <jobid> for more details
