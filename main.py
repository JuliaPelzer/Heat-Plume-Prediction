from torch.nn import MSELoss

import learn_process as lp
from networks.unet_leiterrl import TurbNetG, UNet
from networks.dummy_network import DummyNet
import visualization.visualize_data as vis

# parameters of model and training
loss_fn = MSELoss()
n_epochs = 1 #60000
lr=0.0004 #0.0004

# model choice
#model = TurbNetG(channelExponent=4, in_channels=4, out_channels=2)
unet_model = UNet(in_channels=5, out_channels=1).float()
# TODO too many in channels for unet?
fc_model = DummyNet().float()
# model.to(device)

# init data
datasets_2D, dataloaders_2D = lp.init_data(reduce_to_2D=True, overfit=True)
# TODO: add 3D data
# TODO : data augmentation, 
# train model
lp.train_model(unet_model, dataloaders_2D, loss_fn, n_epochs, lr)
# visualize results, pic in folder visualization/pics under plot_y_exemplary
vis.plot_exemplary_learned_result(unet_model, dataloaders_2D, name_pic="plot_y_exemplary")

# print(dataloaders_2D["train"].dataset[0]['x'].shape)