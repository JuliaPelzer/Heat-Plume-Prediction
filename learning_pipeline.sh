 #!/bin/bash         

destination_name="80_80_3"
dataset_name="dataset"
device_name="cuda:2"
epochs="20000"


python main.py --dataset_raw ${dataset_name} --dataset_prep "${dataset_name}_len_128" --destination "${destination_name}_1" --epochs ${epochs} --loss physical --device ${device_name}

python main.py --dataset_raw ${dataset_name} --dataset_prep "${dataset_name}_ox_len_128" --destination "${destination_name}_2" --epochs ${epochs} --loss physical --device ${device_name} --model "${destination_name}_1" --case finetune

python main.py --dataset_raw ${dataset_name} --dataset_prep "test_ox_len_128" --destination "${destination_name}_test" --device ${device_name}  --model "${destination_name}_2" --case "test"