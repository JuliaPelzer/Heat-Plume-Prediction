python main.py --device cuda:3 --inputs g --problem extend2 --visu True --epochs 50 --dataset_raw dataset_medium_k_3e-10_1000dp --len_box 128 --skip 64 --notes "compare inputs, with first box - for comparability"
python main.py --device cuda:3 --inputs p --problem extend2 --visu True --epochs 50 --dataset_raw dataset_medium_k_3e-10_1000dp --len_box 128 --skip 64 --notes "compare inputs, with first box - for comparability"
python main.py --device cuda:3 --inputs k --problem extend2 --visu True --epochs 50 --dataset_raw dataset_medium_k_3e-10_1000dp --len_box 128 --skip 64 --notes "compare inputs, with first box - for comparability"
# python main.py --device cuda:3 --inputs gk --problem extend2 --visu True --epochs 1000 --dataset_raw dataset_medium_k_3e-10_1000dp --len_box 128 --skip 64 --notes "compare inputs, padding=halfpad" --destination "dataset_medium_k_3e-10_1000dp inputs_gk case_train box128 skip16 e1000"

# python main.py --device cuda:3 --inputs pk --problem extend2 --visu True --epochs 1000 --dataset_raw dataset_medium_k_3e-10_1000dp --len_box 128 --skip 64 --notes "compare inputs, padding=halfpad" --destination "dataset_medium_k_3e-10_1000dp inputs_pk case_train box128 skip16 e1000"

# python main.py --device cuda:3 --inputs gks --problem extend2 --visu True --epochs 1000 --dataset_raw dataset_medium_k_3e-10_1000dp --len_box 128 --skip 64 --notes "compare inputs, padding=halfpad" --destination "dataset_medium_k_3e-10_1000dp inputs_gks case_train box128 skip64 e1000"

# python main.py --device cuda:3 --inputs pks --problem extend2 --visu True --epochs 1000 --dataset_raw dataset_medium_k_3e-10_1000dp --len_box 128 --skip 64 --notes "compare inputs, padding=halfpad" --destination "dataset_medium_k_3e-10_1000dp inputs_pks case_train box128 skip64 e1000"


# python main.py --device cuda:3 --inputs gksi --problem extend2 --visu True --epochs 1000 --dataset_raw dataset_medium_k_3e-10_1000dp --len_box 128 --skip 64 --notes "compare inputs, padding=halfpad horizontal" --destination "dataset_medium_k_3e-10_1000dp inputs_gksi case_train box128 skip64 e1000"

# python main.py --device cuda:3 --inputs pksi --problem extend2 --visu True --epochs 1000 --dataset_raw dataset_medium_k_3e-10_1000dp --len_box 128 --skip 64 --notes "compare inputs, padding=halfpad" --destination "dataset_medium_k_3e-10_1000dp inputs_pksi case_train box128 skip64 e1000"