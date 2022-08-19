from torch.nn import MSELoss

import learn_process as lp
from networks.unet_leiterrl import TurbNetG, UNet
from networks.dummy_network import DummyNet
import visualization.visualize_data as vis
import datetime as dt
import sys
import logging

logging.basicConfig(level=logging.INFO)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL

# parameters of model and training
loss_fn = MSELoss()
n_epochs = 2000 #60000
lr=0.0004 #0.0004

# model options
#model = TurbNetG(channelExponent=4, in_channels=4, out_channels=2)
unet_model = UNet(in_channels=5, out_channels=1).float()
# TODO too many in channels for unet?
fc_model = DummyNet(in_channels=2, out_channels=1).float()

# model choice 
model_choice = sys.argv[1]
if model_choice == "unet":
    model = unet_model
elif model_choice == "fc":
    model = fc_model
else:
    print("model choice not recognized")
    sys.exit()

# model.to(device)

# init data
datasets_2D, dataloaders_2D = lp.init_data(reduce_to_2D=True, overfit=True, dataset_name="groundtruth_hps_no_hps/groundtruth_hps")
print(len(datasets_2D["test"]))

# TODO overfit until not possible anymore (dummynet, then unet)
# therefor: try to exclude connections from unet until it overfits properly (loss=0)
# TODO go over input properties (delete some, some from other simulation with no hps?)
# TODO: add 3D data
# TODO : data augmentation, 
# train model
#lp.train_model(model, dataloaders_2D, loss_fn, n_epochs, lr)
# visualize results, pic in folder visualization/pics under plot_y_exemplary
# current date and time
now = dt.datetime.now()
#vis.plot_exemplary_learned_result(model, dataloaders_2D, name_pic=f"plot_y_exemplary_{now}")

# print(dataloaders_2D["train"].dataset[0]['x'].shape)