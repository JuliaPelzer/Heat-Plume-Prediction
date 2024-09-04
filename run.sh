#!/bin/bash
#SBATCH --job-name=1HP_NN_RUN_EVAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=simcl1n1
#SBATCH --time=10:00

#module load cuda/12.2.2
source /import/sgs.scratch/miliczpl/cnn_env/bin/activate

python main.py --dataset_raw dataset_square_1000dp_p_lurd --data_n 100 --inputs pksi --augmentation_n 0 --case test --model '/import/sgs.scratch/miliczpl/models/100dp/lurd/dataset_square_1000dp_p_lurd restricted_100 inputs_pksi rotate_inputs_ 0 mask_False case_train augmentation_n_0 rotate_inference_False use_ecnn_False'