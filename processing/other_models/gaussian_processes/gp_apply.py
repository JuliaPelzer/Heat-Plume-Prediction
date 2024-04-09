import matplotlib.pyplot as plt
import numpy as np
from data.utils import SettingsTraining
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from main import init_data

print("Loading data...")
args = {"datasets_path": "datasets_prepared",
        "dataset_name": "benchmark_dataset_2d_100datapoints",
        "device": "cuda:3",
        "epochs": 1,
        "finetune": False,
        "path_to_model": "none",
        "model_choice": "unet",
        "name_folder_destination": "default"
        }

settings = SettingsTraining(**args)
datasets, dataloaders = init_data(settings)

train_inputs =  np.array([x for (x,y) in dataloaders["train"]][0])
train_outputs = np.array([y for (x,y) in dataloaders["train"]][0])
val_inputs =    np.array([x for (x,y) in dataloaders["val"]][0])
val_outputs =   np.array([y for (x,y) in dataloaders["val"]][0])

# train_inputs_vector = train_inputs.reshape(4, 256*20*70).T
# train_outputs_vector = train_outputs.reshape(1, 256*20*70).T
# val_inputs_vector = val_inputs.reshape(4, 256*20*20).T
# val_outputs_vector = val_outputs.reshape(1, 256*20*20).T

train_inputs_vector = train_inputs[0].reshape(4, 256*20)
indices = [i/len(train_inputs_vector) for i in range(len(train_inputs_vector))]
train_inputs_vector = np.append(train_inputs_vector, indices)
print(train_inputs_vector.shape)
train_inputs_vector = train_inputs_vector.T
train_outputs_vector = train_outputs[0].reshape(1, 256*20).T
train_inputs = train_inputs_vector[0:1000]
train_outputs = train_outputs_vector[0:1000]
val_inputs = train_inputs_vector[1000:10000]
val_outputs = train_outputs_vector[1000:10000]
print("In and output shapes of train and val", train_inputs.shape, train_outputs.shape, val_inputs.shape, val_outputs.shape)

print("Initializing GP...")
kernel = RBF(1.0)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

print("Fitting GP...")
gp = gpr.fit(train_inputs, train_outputs)

print("Predicting GP...")
y_pred, y_pred_std = gp.predict(val_inputs, return_std=True)

# plot
print("Plotting GP...")
plt.plot(val_inputs, y_pred, 'b.', label='Prediction')
plt.plot(train_inputs, train_outputs, 'r*', markersize=10, label='Observations')
plt.fill(np.concatenate([val_inputs, val_inputs[::-1]]),
            np.concatenate([y_pred - 1.9600 * y_pred_std,
                            (y_pred + 1.9600 * y_pred_std)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
# plt.legend(loc='upper left')
plt.savefig('gp.png')
print("Done!")
