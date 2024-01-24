# Goal
Predict the groundwater temperature field near groundwater heat pumps.
- The grounds permeability and pressure gradient is variable over the whole area of interest.
- The heat plumes of multiple heatpums can interact

# Current State

## Approach 1: Two Stage Network

Use a large U-Net to predict the temperature field of a single heatpump, then use a second network (stage) to predict the combined temperature field of multiple heatpumps by correcting for the interaction of the heatplumes.

### Problems:
1. The U-Net has a fixed size, and runtime and parameters scale with the area visible to the network. This leads to long training times for larger fields of view
2. The Field-Of-View Training-Time tradeoff does not allow for arbitrarily large input fields. This becomes a problem when the groundwater flows to fast or does not cool quickly enough, such that the heatplumes would affect an area outside of the field of view of the network
3. Addig a third dimension or time dependence would drastically increase memory and compute requirements to the point where it is not feasable to continue this approach

### Learnings:
- Two stage prediction is feasible
- The combination of two undisturbed heatplumes using a network does produce accurate results.

## Approach 2: All-In-One

All-In-One: All heatpumps and heatplumes get predicted at the same time by one now fully convolutional network. Training and inference are now completely independent from each other. This means it can predict arbitrarily large areas filled with heatpumps, such as the intended usecase of modeling the real world influence of heatpumps on groundwater in a large city such as munich. The infered temperature field can be much larger than the simulated training data.

### Problems:
1. The required network size for good accuracy is even larger than in the two stage approach.
2. The field of view is still fixed and directly related to training time.
3. Adding additional input dimensions exponentially increases the required compute and memory.

### Learnings:
- The All-In-One approach yields good results with an overall simpler pipeline.
- Fully Convolutional Networks allow decoupling of training input size from inference input size.

# Proposal: NeRF-like approach for the first stage of [Approach 1](#approach-1-two-stage-network)

Instead of using the discretized permeability and pressure fields as input (~2x5000 dimensions) and training for an also discretized temperature field (~5000 dimensions), the first stage itself is split into two steps:

## 1. Heatpump specific temperature field

A new network is introduced to model the temperature at any point $\vec{x}$ as 
$$
T(\vec{x}, \dot{m}, T_{out}, \dots) : \mathbb{R}^n \mapsto \mathbb{R}
$$
, depending on only the parameters of the heatpump, such as massflow $\dot{m}$ or outflow temperature $T_{out}$. As this function has only $n\approx 5$ input dimensions and a scalar output, it can be represented by a very small and shallow dense network. The training data for this network consists of variations in the input parameters and fixed, common values for permeability and pressure.

The concept of using a scalar function to model a field is the same as in neural radiance fields (NeRF). The existing research  for faster training, more accurate small scale resolution of NeRFs could be leveraged if nessesary.

Adding a third dimension, time dependence or any other input parameter leads to a linear instead of exponential increase in network size.

A prediction at a single point is now in the order of microseconds, as it does not take any neighboring fields into account.

### Proof of Concept

This general approach can quickly be experimentally verified.

(The network in the picture uses only two-dimensional coordinates and no additional parameters to model the temperature. It has two fully connected layers with only 17k trainable parameters and finishes training in just under two minutes on a laptop.)

![proof of concept](images/NeRF%20poc.png)

This creates a *differentiable* function for predicting the temperature at any given point near the heatpump.

## 2. Learned coordinate transformation

When permeability $K$ or pressure $P$ vary over space, the heatplumes do not follow a straight line anymore, but can be longer, shorter, curved, split or distorted in any other way. This behavior is not modeled in the simple NeRF-like first stage.

If the distortion was known beforehand, the coordinates that are given to the first stage could be distorted correspondingly.
This would allow changes to the NeRF stage without retraining any other network.

The distortion (deflection) is locally dependent on the permeability field and the pressure. This allows the use of a (really small) fully convolutional neural network like in [Approach 2](#approach-2-all-in-one). This network would predict the local distortion at any grid point (of the discrete permeability/pressure field).


The network to predict the local distortion models a function 
$$
\vec{D}(K_{surronding}, P_{surrounding}): \mathbb{R}^k \times \mathbb{R}^k \mapsto \mathbb{R}^d
$$
, where $k\approx 16$ is the number of neighboring cells taken into account, and $d\in[2,3]$ are the spatial dimensions. The output is the scale of the transformed cell in each direction.

The global coordinates (relative to the heatpumps location) $\vec{C}(\vec{x}): \mathbb{R}^d \mapsto \mathbb{R}^d$ can then be constructed by accumulating the local distortions $D$ using a cummulative sum over each spatial axis.

A temperature prediction $P$ is then calculated as 
$$
P(\vec{x},K(\vec{x}),P(\vec{x}),\dot{m},\dots) = T(\vec{C}(\vec{x},K_{surrounding}(\vec{x}),P_{surrounding}(\vec{x})), \dot{m},\dots)
$$
A single temperature point 2D needs the local distortions of all coordinates in a rectangle between itself and the heatpump.

To train this model, the entire prediction pipeline is used with the weights of the NeRF-like stage frozen to keep the temperature network indepenent of the distortion network. A MSE loss between predicted and simulated temperature should give good gradients. 

### On training stability

All parts of the prediction function are differentiable, but it is also important that the NeRF-like temperature function is sufficiently smooth. The global distortion function can only modify the temperature output by modifying the distorted coordinates slightly.

Training could also become unstable as the global distortion function depends on equal parts on all local distortions. 
Some tricks might be nessesary to get usable local distortions, such as:
- Some insights from the "Nerfies - Deformable Neural Radiance Fields" paper
- low distortion in the early stages of training
- handcrafted weight initialization
- loss with weight of a datapoint depending on the distance from the heatpump to avoid error accumulation
