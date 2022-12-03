#!/bin/bash
# CHANGE JOB NAME
#SBATCH --job-name "OCT7"
#SBATCH --output "log_AUTOREGRESSIVE/output_sbatch_Autoregressive_.%j.log"

#SBATCH --nodes=1

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#32
#SBATCH --mem-per-cpu 30
#####  # #SBATCH --time-min=90:90
#SBATCH --oversubscribe
#SBATCH --priority=TOP


#DAVIDSON
export DATA_DIR='/home/DAVIDSON/alalkadhim.visitor/IQN/DAVIDSON_NEW/data'
#LOCAL
#export DATA_DIR='/home/ali/Desktop/Pulled_Github_Repositories/IQN_HEP/Davidson/data'
conda init bash
source ~/.bashrc
conda activate torch_env
conda info --envs

#srun python Train_Evaluate_m.py --n_iterations 8000000 --n_layers 1 --n_hidden 64 --starting_learning_rate 2.1205283263244004e-02
srun python Train_Evaluate_pT.py --n_iterations 8000000 --n_layers 12 --n_hidden 256 --starting_learning_rate 1.0404334991989074e-02
#srun python Train_Evaluate_Eta.py --n_iterations 2000000 --n_layers 6 --n_hidden 64
#srun python Train_Evaluate_phi.py --n_iterations 2000000 --n_layers 6 --n_hidden 64



### DO scontrol -d show job <jobid> for more details
