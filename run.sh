#!/bin/bash
#SBATCH --job-name=1HP_NN_RUN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=simcl1n2
#SBATCH --time=24:00:00

#module load cuda/12.2.2
source /import/sgs.scratch/miliczpl/cnn_env/bin/activate

python main.py --dataset_raw dataset_square_100dp_p_turned90_higher_freq --inputs pksi --augmentation_n 6 --case test --rotate_inference True --model '/import/sgs.scratch/miliczpl/1HP_NN_equivariance/runs/1hpnn/dataset_square_100dp_p_right_higher_freq inputs_pksi case_train augmentation_n_1 box_256 skip_256 rotate_inference_False'