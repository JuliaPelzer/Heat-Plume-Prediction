#!/bin/bash
#SBATCH --job-name=1HP_NN_RUN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=simcl1n2
#SBATCH --time=24:00:00

#module load cuda/12.2.2
source /import/sgs.scratch/miliczpl/cnn_env/bin/activate

python main.py --dataset_raw dataset_square_1000dp_p_random_dir --inputs pksi --case test --augmentation_n -1 --model '/import/sgs.scratch/miliczpl/models/1000dp/random_dir/dataset_square_1000dp_p_random_dir restricted_-1 inputs_pksi rotate_inputs_ 0 mask_False case_train augmentation_n_-1 rotate_inference_False use_ecnn_False'