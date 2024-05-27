import torch
import matplotlib.pyplot as plt
import numpy as np

path = "/home/hofmanja/test_nn/runs/extend_plumes2/dataset_medium_100dp_vary_perm inputs_gksi case_test net_convLSTM steps_20 predictBox_"
path_plot = "/home/hofmanja/test_nn/runs/extend_plumes2/"

steps = 20

for dp in range(8):
    results = []
    labels = []
    perm_labels = []
    perm_preds = []
    for box_nr in range(3,12):
        result_file = f'pred_dp{dp}_box{box_nr}.pt'
        label_file = f'label{dp}_box{box_nr}.pt'
        result = torch.load(f'{path}{box_nr}/{result_file}', map_location=torch.device('cpu'))[4].squeeze()
        label = torch.load(f'{path}{box_nr}/{label_file}', map_location=torch.device('cpu'))[4].squeeze()
        perm_label = torch.load(f'{path}{box_nr}/{label_file}', map_location=torch.device('cpu'))[1].squeeze()
        results.append(result)
        labels.append(label)
        perm_labels.append(perm_label)
    

    predicted_plume = torch.cat(results,dim=0).T
    label = torch.cat(labels, dim=0).T
    perm_label = torch.cat(perm_labels, dim=0).T
    error = predicted_plume - label

    fig, ax = plt.subplots(4, figsize=(9,7))
    im0 = ax[0].imshow(predicted_plume, cmap='hot')
    im1 = ax[1].imshow(label, cmap='hot')
    im2 = ax[2].imshow(perm_label, cmap='viridis')
    im3 = ax[3].imshow(error, cmap='coolwarm')

    cbar = fig.colorbar(im0, ax=ax[0], orientation='vertical')
    cbar = fig.colorbar(im1, ax=ax[1], orientation='vertical')
    cbar = fig.colorbar(im2, ax=ax[2], orientation='vertical')
    cbar_error = fig.colorbar(im3, ax=ax[3], orientation='vertical')
    
    ax[0].set_title('Predicted Plume')
    ax[1].set_title('Label')
    ax[2].set_title('Permeability')
    ax[3].set_title('Error')
    
    plot_name = f'predicted_plume_steps{steps}_dp{dp}_withPerm'
    plt.savefig(f"{path_plot}{plot_name}.png")
    plt.close()



