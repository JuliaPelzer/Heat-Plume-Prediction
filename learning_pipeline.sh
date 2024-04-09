 #!/bin/bash         

destination_name="vary_perm"
testset_name="plume_extension_training_vary_perm"
trainset_name="plume_extension_training_vary_perm"
device_name="cuda:3"
epochs="20000"
len="128"


python main.py --dataset_raw "" --dataset_prep "${trainset_name}_len_${len}" --destination "${destination_name}_1" --epochs ${epochs} --loss physical --device ${device_name} --visualize False

python main.py --dataset_raw "" --dataset_prep "${trainset_name}_len_${len}" --destination "${destination_name}_1_vis_train_short" --device ${device_name}  --model "${destination_name}_1" --case "test"

python main.py --dataset_raw "" --dataset_prep "${trainset_name}_ox_len_${len}" --destination "${destination_name}_1_vis_train_long" --device ${device_name}  --model "${destination_name}_1" --case "test"

python main.py --dataset_raw "" --dataset_prep "${testset_name}_len_${len}" --destination "${destination_name}_1_vis_test_short" --device ${device_name}  --model "${destination_name}_1" --case "test"

python main.py --dataset_raw "" --dataset_prep "${testset_name}_ox_len_${len}" --destination "${destination_name}_1_vis_test_long" --device ${device_name}  --model "${destination_name}_1" --case "test"

python main.py --dataset_raw "" --dataset_prep "${trainset_name}_len_${len}" --destination "${destination_name}_2" --epochs ${epochs} --loss physical --device ${device_name} --model "${destination_name}_1" --case finetune --visualize False --number_it 6

python main.py --dataset_raw "" --dataset_prep "${trainset_name}_len_${len}" --destination "${destination_name}_3" --epochs ${epochs} --loss physical --device ${device_name} --model "${destination_name}_2" --case finetune --visualize False --number_it 12

python main.py --dataset_raw "" --dataset_prep "${trainset_name}_len_${len}" --destination "${destination_name}_f" --epochs ${epochs} --loss physical --device ${device_name} --model "${destination_name}_3" --case finetune --visualize False --number_it 18

#python main.py --dataset_raw "" --dataset_prep "${trainset_name}_len_${len}" --destination "${destination_name}_f" --epochs ${epochs} --loss physical --device ${device_name} --model "${destination_name}_4" --case finetune --visualize False --number_it 24

python main.py --dataset_raw "" --dataset_prep "${trainset_name}_len_${len}" --destination "${destination_name}_f_vis_train_short" --device ${device_name}  --model "${destination_name}_f" --case "test"

python main.py --dataset_raw "" --dataset_prep "${trainset_name}_ox_len_${len}" --destination "${destination_name}_f_vis_train_long" --device ${device_name}  --model "${destination_name}_f" --case "test"

python main.py --dataset_raw "" --dataset_prep "${testset_name}_len_${len}" --destination "${destination_name}_f_vis_test_short" --device ${device_name}  --model "${destination_name}_f" --case "test"

python main.py --dataset_raw "" --dataset_prep "${testset_name}_ox_len_${len}" --destination "${destination_name}_f_vis_test_long" --device ${device_name}  --model "${destination_name}_f" --case "test"