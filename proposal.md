# Goal
Predict the groundwater temperature field near groundwater heat pumps.
- The subsurface pressure gradient and permeability are variable over the whole area of interest.
- The heat plumes of multiple heat pumps can interact

# Current State

## Approach 1: Two Stage Network

Use a large U-Net to predict the temperature field of a single heat pump, then use a second network (stage) to predict the combined temperature field of multiple heat pumps by correcting the output of the first stage for the interaction of the heat plumes.

### Challenges:
1. The U-Net has a fixed size, and runtime and parameters scale with the area visible to the network. This leads to long training times for larger fields of view. The Field-Of-View Training-Time tradeoff does not allow for arbitrarily large input fields. This becomes a problem when the groundwater flows too fast and does not cool quickly enough, such that the heat plumes would extend outside of the field of view of the network
2. Adding a third dimension or time dependence would drastically increase memory and compute requirements to the point where it is not feasible to continue this approach.

### Learnings:
- Two stage prediction is feasible.
- The combination of two undisturbed heat plumes using a network does produce physically accurate results.

## Approach 2: All-In-One

All-In-One: All heat pumps and heat plumes of a larger domain are predicted at the same time by one fully convolutional network. Training and inference are now completely independent of each other. This means the network can predict arbitrary large areas with lots of heat pumps, even when trained on much smaller domains. This is necessary for the intended usecase of modeling the real world influence of heat pumps on groundwater in a large city such as Munich. The inferred temperature field can be much larger than the simulated training data.

### Challenges:
1. The required network size for good accuracy is even larger than in the two stage approach.
2. The field of view is still fixed and directly related to training time.
3. Adding additional input dimensions exponentially increases the required compute and memory.

### Learnings:
- The All-In-One approach yields good results with an overall simpler pipeline.
- Fully Convolutional Networks allow decoupling of training input size from inference input size.

# Proposal: NeRF-like approach for the first stage of [Approach 1](#approach-1-two-stage-network)

Instead of using the discretized permeability and pressure fields as input (2x16x256 (~8000) dimensions) and training for an also discretized temperature field (~4000 dimensions), the first stage itself is split into two steps:

## 1. step: Heat pump specific temperature field

A new network is introduced to model the temperature at any position $\vec{x}\mathbb{R}^2$ in the domain as 
$$T(\vec{x}, Q, T_{inj}, \dots) : \mathbb{R}^n \mapsto \mathbb{R},$$
depending only on the parameters of the heat pump, such as volumetric pump rate $Q$ and injection temperature $T_{inj}$. As this function has only $n\approx 4$ input dimensions and a scalar output, it can be represented by a very small and shallow dense network. The training data for this network consists of variations in the input parameters of the heat pump named above and fixed standard values for the subsurface parameters such as permeability and pressure gradient.

The concept of using a scalar function to model a continuous field is the same as in neural radiance fields (NeRF). The existing research  for faster training, more accurate small scale resolution of NeRFs could be leveraged if necessary.

Adding a third dimension, time dependence or any other input parameter leads to a linear instead of exponential increase in network size.

A prediction at a single point $\vec{x}$ is in the order of microseconds.

### Proof of Concept

This general approach can quickly be experimentally verified by overfitting on one data point with fixed heat pump parameters.

(The network in the picture uses only two-dimensional coordinates and no additional parameters to model the temperature. It has two fully connected layers with only 17k trainable parameters and finishes training in just under two minutes on a laptop.)

![proof of concept](images/NeRF%20poc.png)

This creates a *differentiable* function for predicting the temperature at any given point near the heat pump.

## 2. step: Coordinate transformation

When the permeability $k$ or the pressure gradient $\nabla p$ vary over space, the heat plumes do not follow a straight line anymore, but can be longer, shorter, curved, split or distorted in any other way. This behavior is not modeled in the simple NeRF-like first step.

If this distortion were modeled beforehand, the coordinates that are given to the NeRF-like step could be distorted correspondingly.
This would allow changes to the NeRF step without retraining any other network.

The distortion (deflection) is locally dependent on the permeability field and the pressure gradient. This allows the use of a small, fully convolutional neural network such as in [Approach 2](#approach-2-all-in-one). This network predicts the local distortion at any grid point (of the discrete permeability/pressure field).

The network to predict the local distortion models a function 

$$
\vec{D}(\vec{x}) = \vec{D}(k_{surrounding}, p_{surrounding}): \mathbb{R}^k \times \mathbb{R}^k \mapsto \mathbb{R}^d,
$$

where $k\approx 16$ is the number of surrounding cells taken into account, and $d\in[2,3]$ are the spatial dimensions. The output is the size of the transformed cell in each direction. 

The global coordinates (relative to the heat pump's location) $\vec{C}(\vec{x}): \mathbb{R}^d \mapsto \mathbb{R}^d$ can then be constructed by a path integral over the local distortions $D$ from the heatpump to the untransformed coordinate:
$$ \vec{C}(\vec{x}) = \oint_{\vec{x}_{heatpump}}^{\vec{x}} D(\vec{x}) d\vec{x}'$$
The discrete case is just a sum over the local distortions.

Consistent global coordinates are only possible if the value of the integral is independent of the path taken, implying zero curl of the distortion field ($\nabla \times \vec{D} = \vec{0}$) and requiering the local distortions to be a gradient field $\vec{D} = \nabla F$ (of an arbitrary field $F$). 

### Enforcing the constraints on the local distortion
The only way to mathematically ensure that $\vec{D}$ is a conservative field is to learn the global field $F$ and taking the gradient. This approach would not allow the use of small local distortion predictions, drastically increasing the required network size to the point where [Approach 1](#approach-1-two-stage-network) would be more efficient.

Possible ways to still get a conservative field are:
1. Always use the same easy to calculate paths (e.g. suitable for cummulative sum) and hope and pray that the network learns a reasonable field
2. Add a loss that penalizes local curl
3. Always take a random path integral, forcing the network to learn a conservative field to maintain consistent results for the same input

### Training

A temperature prediction $P$ **P for T_pred?** is then calculated as 
$$P(\vec{x},k(\vec{x}),p(\vec{x}),Q,\dots) = T(\vec{C}(\vec{x},k_{surrounding}(\vec{x}),p_{surrounding}(\vec{x})), Q,\dots).$$
**bisschen unverständlich was genau du hier meinst**

To train this model, the entire prediction pipeline is used with the weights of the NeRF-like step frozen to keep the temperature network independent of the distortion network. A MSE loss between predicted and simulated temperature should give good gradients. 

#### On training stability

All parts of the prediction function are differentiable, but it is also important that the NeRF-like temperature function is sufficiently smooth. The global distortion function can only modify the temperature output by modifying the distorted coordinates slightly.

Training could also become unstable as the global distortion function depends on equal parts on all local distortions. 
Some tricks might be necessary to get stable local distortions, such as:
- Some insights from the "Nerfies - Deformable Neural Radiance Fields" paper
- low distortion in the early stages of training
- handcrafted weight initialization
- loss with weight of a datapoint depending on the distance from the heat pump to avoid error accumulation
