#!/bin/bash
#SBATCH --job-name=1HP_NN_RUN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=simcl1n2
#SBATCH --time=24:00:00

#module load cuda/12.2.2
source /import/sgs.scratch/miliczpl/cnn_env/bin/activate

python main_hyperparam.py --mask True --dataset_raw dataset_square_1000dp_p_random_dir --inputs pksi --augmentation_n 4 --epochs 1000