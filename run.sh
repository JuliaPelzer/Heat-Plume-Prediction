#!/bin/bash
#SBATCH --job-name=1HP_NN_RUN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=simcl1n1
#SBATCH --time=24:00:00

#module load cuda/12.2.2
source /import/sgs.scratch/miliczpl/cnn_env/bin/activate

python main.py --dataset_raw dataset_square_100dp_p_right_higher_freq --inputs pksi --use_ecnn True