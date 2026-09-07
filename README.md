# Time-aware ML-Based Modeling of Heat Flow around Groundwater Heat Pumps

## Abstract

This project introduces a hybrid machine learning and physics-based approach to model heat plumes in groundwater systems. Building upon the **LGCNN** architecture, we extend the methodology to account for time-dependent (seasonal) factors and probabilistic streamline estimation. The method effectively combines Convolutional Neural Networks (CNNs) for local heterogeneity with explicit numerical solvers for global flow dependencies.


## Methodology

The pipeline has three ordered steps that together predict the temperature field $T(\mathbf{x})$. Configured via YAML, a single run can train or evaluate any subset of these steps.

### 1. Local Velocity Estimation
A U-Net predicts the heterogeneous groundwater velocity field $\mathbf{v}(\mathbf{x})$
- inputs: Hydraulic pressure gradient $\nabla p$, Permeability field $k$, Heat pump locations $i$
- outputs: X-velocity, Y-velocity

### 2. Global Streamline Computation
Physics-based stochastic streamlines plus an RWPT thermal prior (not a full PFLOTRAN run). The injection rate in the RWPT solver is scaled by `0.25` to align with PFLOTRAN ground truth.

| Channel | Name | Description |
|---------|------|-------------|
| `1` | Sum Position | Accumulated streamline density |
| `2` | Relative Uncertainty | Std/mean of ensemble |
| `3` | Time-Faded Position | Temporal decay weighting |
| `4` | Max Time-Faded | Peak influence over time |
| `5` | Seasonal Position | Multi-season aggregation |
| `6` | Max Seasonal | Peak seasonal influence |
| `7` | RWPT | GPU-computed thermal plume |

- inputs: X-velocity, Y-velocity, heat-pump positions
- outputs: 7 tensors

### 3. Temperature Regression
A second CNN maps physical priors (and selected raw channels) to $T(\mathbf{x})$.


## Installation

### Prerequisites
* Python >=3.12
* NVIDIA GPU + CUDA recommended for step 2. CPU works for install and light runs; set `run_configuration.device` to `cpu` in the YAML if needed.

### Setup
```sh
# HPC (optional): module load python/3.12.9
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Dataset Setup
~76 GB from DaRUS; expect hours depending on network. If the dataset is still private, put the API key in `.darus_apikey` at the **repo root**. Submodule deps are only in `darus_data_download/requirements.txt`. Re-running `get_data.py` will continue the download.

```sh
git submodule update --init --recursive --force
cd darus_data_download
pip install -r requirements.txt
python scripts/get_data.py
unzip scripts/data/datasets-100-heat-pumps-synthetic-permeability-fields-simulation-raw-3-1-data-points/random_perm_3dp.zip
mkdir -p ../datasets ../results
mv dataset_giant_100hp_varyK ../datasets/
mv scripts/data/student-thesis-on-transient-adaptation-of-lgcnn/datasets/* ../datasets/
mv scripts/data/student-thesis-on-transient-adaptation-of-lgcnn/probabilistic-lgcnn/results/* ../results/
cd ..
```

Expected layout: `datasets/dataset_giant_100hp_varyK`, `datasets/paper-results-seasonal`, and pretrained runs under `results/`. Paper configs use `./datasets` and `./results`.


## Usage

All phases (clean / train / test / step2) are driven by one YAML file and the same entry point:

```bash
python -m code [config_file.yaml]
```

`run_configuration.pipeline` is an ordered list of actions (`clean`, `step1`, `step2`, `step3`). Modes for the CNN steps are typically `train` or `test`; step2 only supports `run`. One config can run the full three-step sequence or isolate a single step.

**Paper configs** (default to **train** for step1/step3):

```sh
python -m code settings/steady-state.yaml   # dataset_giant_100hp_varyK
python -m code settings/seasonal.yaml       # paper-results-seasonal
```

Smoke / minimal configs: `settings/steady-state-test.yaml`, `settings/seasonal-test.yaml`.

**Example pipeline snippet:**

```yaml
run_configuration:
  run_name: seasonal
  dataset: paper-results-seasonal
  device: cuda:0

  pipeline:
    - clean: data_prep
    - step1: train
    - step2: run
    - step3: train

general_configuration:
  step3:
    model_parameters:
      network: unet
      inputs: [p, k, i, x, y, 1, 2, 3, 5, 7]  # 1-7 = streamline / RWPT priors
      outputs: [t]
```

Full option list: `settings/template.yaml`.

### Tests

```sh
python -m unittest discover -s tests
```

Covers config parse, step2 APIs / tiny RWPT, U-Net train+infer, and minimal seasonal / steady-state pipelines via `settings/seasonal-test.yaml` and `settings/steady-state-test.yaml`. Seasonal smoke runs step1→2→3 on CPU; steady-state smoke runs step1 only on CPU (2560^2 / 100 HPs, full step2/3 needs CUDA). Pipeline tests expect the DaRUS datasets under `datasets/`.

### VampireMan data conversion

Upstream data come from [VampireMan](https://github.com/JuliaPelzer/VampireMan) (PFLOTRAN). To convert a VampireMan dataset into this pipeline's layout:

```bash
python code/converter.py [path_to_vampireman_dataset] [path_to_save_dataset]
```


## Acknowledgements

This project is based on the research from [Heat-Plume-Prediction](https://github.com/JuliaPelzer/Heat-Plume-Prediction) and utilizes the [VampireMan](https://github.com/JuliaPelzer/VampireMan) data generation tool.
