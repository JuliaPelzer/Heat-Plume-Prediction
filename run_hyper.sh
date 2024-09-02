#!/bin/bash
#SBATCH --job-name=1HP_NN_RUN_TRAIN
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=simcl1n1
#SBATCH --time=48:00:00

#module load cuda/12.2.2
source /import/sgs.scratch/miliczpl/cnn_env/bin/activate

python main_hyperparam.py --dataset_raw dataset_square_1000dp_p_random_dir --rotate_inference True --data_n 100 --inputs pksi --epochs 10000