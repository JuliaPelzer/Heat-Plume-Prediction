# Comparison of different input combinations
Train and compare on datapoint 4_4 (5years),
validated on datapoint 5_4 (5years)
with skip per direction of 32 and a box-size of 64, resulting in around 3000 datapoints

## Goal Temperature Field 
with input permeability field for reference on the right

<img src="comparison_inputs_on_44/model_overfit_44/train_0_t_true.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_overfit_44/train_0_Permeability X [m^2].png" alt="drawing" width="800"/>

# Predicted (train) Temperature Fields + Error Fields

1. inputs: lik
<img src="comparison_inputs_on_44/model_train44_val54_lik/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_lik/train_0_error.png" alt="drawing" width="800"/>

2. inputs: lmik
<img src="comparison_inputs_on_44/model_train44_val54_lmik/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_lmik/train_0_error.png" alt="drawing" width="800"/>

3. inputs: lmk
<img src="comparison_inputs_on_44/model_train44_val54_lmk/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_lmk/train_0_error.png" alt="drawing" width="800"/>

4. inputs: lmikp
<img src="comparison_inputs_on_44/model_train44_val54_lmikp/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_lmikp/train_0_error.png" alt="drawing" width="800"/>

5. inputs: mik
<img src="comparison_inputs_on_44/model_train44_val54_mik/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_mik/train_0_error.png" alt="drawing" width="800"/>

6. inputs: mikp
<img src="comparison_inputs_on_44/model_train44_val54_mikp/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_mikp/train_0_error.png" alt="drawing" width="800"/>

7. inputs: lmkp
<img src="comparison_inputs_on_44/model_train44_val54_lmkp/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_lmkp/train_0_error.png" alt="drawing" width="800"/>

## Conclusion
- l is definitely required to get the extended lengths of plumes! functionality can not be replaced by p; but it also makes it a bit fuzzy
- including i is causing harm: it does not change the general shape but increases the maximum temperature
- including p helps to get the direction of the plume right more often, also it is clearer by less often splitting as if uncertain
- --> **use lmikp as input for now**

Since lmikp is currently the best, lets look at the respective validation results:

<img src="comparison_inputs_on_44/model_train44_val54_lmikp/test_0_t_true.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_lmikp/test_0_Permeability X [m^2].png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_lmikp/test_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_inputs_on_44/model_train44_val54_lmikp/test_0_error.png" alt="drawing" width="800"/>

# Comparison of different box sizes and skip per direction
skips = [4, 8, 16, 32, 64]
box_sizes = [32, 64, 128]

1. box size 32, skips = [4, 8, 16, 32, 64]

<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip4 box32/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip4 box32/train_0_error.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box32/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip16 box32/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip32 box32/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip64 box32/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip64 box32/train_0_error.png" alt="drawing" width="800"/>

2. box size 64, skips = [4, 8, 16, 32, 64]

<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip4 box64/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip4 box64/train_0_error.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box64/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip16 box64/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip32 box64/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip64 box64/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip64 box64/train_0_error.png" alt="drawing" width="800"/>


3. box size 128, skips = [4, 8, 16, 32, 64]

<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip4 box128/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip4 box128/train_0_error.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box128/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip16 box128/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip32 box128/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip64 box128/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip64 box128/train_0_error.png" alt="drawing" width="800"/>

## Conclusion
- less skip results in too high max-Temp but it is better capable of predicting the plume directions, we see that in all 3 box-sizes-series - maybe just because of more datapoints?

# Compare different box sizes with fixed skip of 8
skips = [8]
box_sizes = [32, 64, 128]

1. box size 32, skips = [8]

<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box32/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box32/train_0_error.png" alt="drawing" width="800"/>

2. box size 64, skips = [8]

<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box64/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box64/train_0_error.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box64/train_0_t_true.png" alt="drawing" width="800"/>

3. box size 128, skips = [8]

<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box128/train_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box128/train_0_error.png" alt="drawing" width="800"/>


## Conclusion
- box 64 (middle one) shows the highest absolute temperatures but regarding the error field it is visible that the maximum error is the lowest - leading to the conclusion that this model is predicting the shapes the best and requires more finetuning
- maybe because of trade off between more datapoints and larger receptive field?

- --> **use box size 64 and skip 8**
- --> **wait for more results on smaller skips for box64**

# Check box 64 with skip 8 on validation data
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box64/test_0_t_out.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box64/test_0_error.png" alt="drawing" width="800"/>
<img src="comparison_boxsizes_on_44/model_train44_val54_lmikp skip8 box64/test_0_t_true.png" alt="drawing" width="800"/>

## Conclusion
- not generalizing at all so far

# Idea: validate on whole 5_4 instead of cutouts:

train

<img src="model_train44_val54Large_lmikp skip8 box64/train_0_t_out.png" alt="drawing" width="800"/>
<img src="model_train44_val54Large_lmikp skip8 box64/train_0_error.png" alt="drawing" width="800"/>

val/test

<img src="model_train44_val54Large_lmikp skip8 box64/val_0_t_out.png" alt="drawing" width="800"/>
<img src="model_train44_val54Large_lmikp skip8 box64/val_0_error.png" alt="drawing" width="800"/>

compared to the cutout validation above: not better: immediatly overfits (see loss plot)


# Number of datapoints in training (and in validation) set

|box\skip | **4** | **8** | **16** | **32** | **64** |
|---|---|---|----|----|----|
| **32** | 242064 | 60516 | 15129 | 3782 | 945 |
| **64** | 234256 | 58564 | 14641 | 3660 | 915 |
| **128** | ? | 54756 | 13689 | 3422 | 855 |

# IMPORTANT NOTE: ALL DONE ON 5 years!! Change back in prepare_1ststage.py