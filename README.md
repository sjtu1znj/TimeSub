# TimeSub: A Time-varying Subdistribution Hazard Model with Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/TimeSub)](https://pypi.org/project/TimeSub/)

A Python package for survival analysis with competing risks, integrating neural networks and statistical inference. Designed for modeling time-varying subdistribution hazards and performing hypothesis testing.

**Corresponding Paper**:  
*A Time-varying Subdistribution Hazard Model with Neural Network*  

---

## Features
- 🧠 **Neural Network Integration**: Flexible architectures for time-varying and non-time-varying subdistribution hazard modeling.
- ⚡ **Competing Risks Support**: Handles right-censored data with multiple event types.
- 📊 **Model Evaluation**: Compute time-dependent AUC and generalized C-index for predictive performance.
- 🔬 **Hypothesis Testing**: Bootstrap-based structure and significance tests for model validation.
- 🚀 **GPU Acceleration**: PyTorch backend for efficient training on CUDA-enabled devices.

---



## Installation
```bash
pip install TimeSub
```
---

## Usage Example

This section provides a practical example of how to use the `TimeSub` package for training various subdistribution hazard models, performing structure and significance tests, and evaluating prediction ability.

First, import the necessary modules:

```python
from TimeSub.estimation import cr_data, create_net_class
from TimeSub.prediction import prediction_ability
import numpy as np
import pandas as pd
```

Define time breaks for time-varying covariates (if applicable):

```python
time_breaks = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
```

### Prepare Training Data

Load training data and format it for `TimeSub`. This example uses time-invariant (TI) covariates.

```python
cov = np.load("data/covariates_TI_train.npy")
data = pd.read_csv("data/sample_TI_train.csv")
# For time-varying (TV) covariates, uncomment the lines below:
# cov = np.load("data/covariates_TV_train.npy")
# data = pd.read_csv("data/sample_TV_train.csv")

t_vec = data['T']
T = data['T'].to_numpy().reshape(-1, 1)
interval_indices = np.searchsorted(time_breaks, t_vec, side='right')-1
Z = cov
n = Z.shape[0]
d = Z.shape[1]

TX = np.zeros((n, n, d+1))
TX[:,:,0] = T
for i in range(d):
    TX[:, :, i+1] = cov[:, i].reshape(1, n)
# For time-varying covariates, uncomment the lines below and adjust 'cov':
# Z = cov[:,interval_indices,:].transpose(1, 0, 2)
# TX = np.zeros((Z.shape[0], Z.shape[1], Z.shape[2]+1))
# TX[:,:,0] = T
# TX[:, :, 1:] = Z

Delta = (data['event_type']>0).to_numpy(dtype=float).reshape(-1, 1)
epsilon = (data['event_type']).to_numpy(dtype=float).reshape(-1, 1)

data_train = cr_data(T, Delta, epsilon, Z, TX, cov)
data_train.to_torch()
```

### Prepare Test Data

Similarly, prepare the test data.

```python
cov = np.load("data/covariates_TI_test.npy")
data = pd.read_csv("data/sample_TI_test.csv")
# For time-varying (TV) covariates, uncomment the lines below:
# cov = np.load("data/covariates_TV_test.npy")
# data = pd.read_csv("data/sample_TV_test.csv")

t_vec = data['T']
T = data['T'].to_numpy().reshape(-1, 1)
interval_indices = np.searchsorted(time_breaks, t_vec, side='right')-1
Z = cov
n = Z.shape[0]
d = Z.shape[1]

TX = np.zeros((n, n, d+1))
TX[:,:,0] = T
for i in range(d):
    TX[:, :, i+1] = cov[:, i].reshape(1, n)
# For time-varying covariates, uncomment the lines below and adjust 'cov':
# Z = cov[:,interval_indices,:].transpose(1, 0, 2)
# TX = np.zeros((Z.shape[0], Z.shape[1], Z.shape[2]1))
# TX[:,:,0] = T
# TX[:, :, 1:] = Z

Delta = (data['event_type']>0).to_numpy(dtype=float).reshape(-1, 1)
epsilon = (data['event_type']).to_numpy(dtype=float).reshape(-1, 1)

data_test = cr_data(T, Delta, epsilon, Z, TX, cov)
data_test.to_torch()
```

### Structure Test

Perform a structure test to assess if a time-varying model significantly improves fit over a time-invariant model.

```python
data_train.structure_test(B_seeds = [100000+(i+1)*10 for i in range(100)],
                          learning_rate = 1e-3, 
                          num_epoch = 300, 
                          num_epoch_ = 50, 
                          num_epoch_B = 50, 
                          learning_rate_B = 5e-2, 
                          Nets = [create_net_class(6, 12, 3)], # Main model (e.g., TNSHM)
                          max_batch_size = 2000 ,
                          valid_rate = 0.5,
                          Nets_TI = [create_net_class(5, 12, 3)], # Null model (e.g., NSHM)
                          log = None,
                          num_model = 5, 
                          num_model_B = 5)
# plr: statistic value.
# time: time taken for the structure test.
# plr_B_time_list: list of statistic values from bootstrap samples.
plr, time = data_train.plr_st
plr_B_time_list = data_train.plr_st_B
```

**Parameters for `structure_test`**:
- `B_seeds`: list of bootstrap seeds.
- `learning_rate`: learning rate for initial model training.
- `num_epoch`: number of epochs for initial model training.
- `num_epoch_`: number of epochs for model training during test statistic calculation.
- `num_epoch_B`: number of epochs for model training during bootstrap test statistic calculation.
- `learning_rate_B`: learning rate for model training during test statistic calculation.
- `Nets`: list of neural network classes for the main (alternative) model. E.g., `create_net_class(6, 12, 3)` means input size 6, hidden size 12, depth 3.
- `max_batch_size`: maximum batch size for training.
- `valid_rate`: proportion of data used for validation.
- `Nets_TI`: list of neural network classes for the null model (time-invariant).
- `log`: file path for logging results.
- `num_model`: number of random initializations for model training during test statistic calculation.
- `num_model_B`: number of random initializations for model training during bootstrap test statistic calculation.

### Train Various Models and Evaluate Prediction Ability

Train different subdistribution hazard models (SHM, TSHM, NSHM, TNSHM) and evaluate their predictive performance using the Generalized C-index (GC) and time-dependent AUC.

```python
data_train.linear(num_epoch = 200, 
                  learning_rate = 0.01)
# The original subdistribution hazard model (SHM).
# num_epoch: number of epochs for training.
# learning_rate: learning rate for optimization.

data_train.spline(num_spline = 10,
               num_epoch = 500,
               learning_rate = 0.01)
# The spline-based time-varying coefficient subdistribution hazard model (TSHM) using cubic splines.
# num_spline: number of spline basis functions.
# num_epoch: number of epochs for training.
# learning_rate: learning rate for optimization.

data_train.nn_ti(Nets = [create_net_class(5, 10, 1), create_net_class(5, 20, 3)],
            batch_num = 1,
            learning_rate1=[1e-2, 1e-3],
            random_num = 5,
            num_epoch = 300)
# The nonparametric subdistribution hazard model (NSHM) using neural networks.
# Nets: list of neural network classes to be used.
# batch_num: number of batches for training.
# learning_rate1: list of learning rates for different stages of training. Corresponds to Nets.
# random_num: number of random initializations for model selection.
# num_epoch: number of epochs for training.

data_train.nn_tv(Nets = [create_net_class(6, 10, 1), create_net_class(6, 20, 3)],
              batch_num = 1,
              learning_rate1=[1e-2, 1e-3],
              random_num = 5,
             num_epoch = 300)
# Time-varying nonparametric subdistribution hazard model (TNSHM) using neural networks.
# Nets: list of neural network classes to be used. E.g., (6,10,1) means input size 6, hidden size 10, depth 1.
# batch_num: number of batches for training.
# learning_rate1: list of learning rates for different stages of training. Corresponds to Nets.
# random_num: number of random initializations for model selection.
# num_epoch: number of epochs for training.

# Predict log subdistribution hazards on the test set
g_torch_linear =  data_train.models['linear'][0](data_test)
g_torch_spline =  data_train.models['spline'][0](data_test)
g_torch_nntv =  data_train.models['nn_tv'][0](data_test)
g_torch_nnti =  data_train.models['nn_ti'][0](data_test)

# g_torch_linear: predicted log subdistribution hazard from SHM.
# g_torch_spline: predicted log subdistribution hazard from TSHM.
# g_torch_nntv: predicted log subdistribution hazard from TNSHM.
# g_torch_nnti: predicted log subdistribution hazard from NSHM.

# Evaluate prediction ability
gc_linear, auc_linear =  prediction_ability(data_test, g_torch_linear, g0_torch=g_torch_nntv) 
gc_spline, auc_spline =  prediction_ability(data_test, g_torch_spline, g0_torch=g_torch_nntv)
gc_nntv, auc_nntv =  prediction_ability(data_test, g_torch_nntv, g0_torch=g_torch_nntv)
gc_nnti, auc_nnti =  prediction_ability(data_test, g_torch_nnti, g0_torch=g_torch_nntv)

# gc_linear: GC for SHM.
# auc_linear: AUC for SHM.
# gc_spline: GC for TSHM.
# auc_spline: AUC for TSHM.
# gc_nntv: GC for TNSHM.
# auc_nntv: AUC for TNSHM.
# gc_nnti: GC for NSHM.
# auc_nnti: AUC for NSHM.
```

### Significance Test

Perform a significance test for a specific covariate to determine if its effect is statistically significant.

**Note**: `num_B` and `learning_B` are placeholders that should be defined. For instance: `num_B = 50`, `learning_B = 5e-2`. `B_seeds` should also be defined, for example, `B_seeds = [100000 + (i + 1) * 10 for i in range(100)]`.

```python
# Example definitions for num_B, learning_B, and B_seeds if not already set:
# num_B = 50
# learning_B = 5e-2
# B_seeds = [100000 + (i + 1) * 10 for i in range(100)]

data_train.significance_test(cov_index = 5, # Index of the covariate to be tested
                             B_seeds=B_seeds, # Use the defined B_seeds
                             learning_rate = 1e-3, 
                             num_epoch = 300, 
                             num_epoch_ = num_B, # Use the defined num_B
                             num_epoch_B = num_B, # Use the defined num_B
                             learning_rate_B = learning_B, # Use the defined learning_B
                             Nets = [create_net_class(6, 12, 3)], # Main model (with covariate)
                             max_batch_size = 2000,
                             valid_rate = 0.5,
                             Nets_null = [create_net_class(5, 12, 3)], # Null model (without covariate)
                             log = None,
                             num_model = 5, 
                             num_model_B = 5)
# plr: statistic value.
# time: time taken for the significance test.
# plr_B_time_list: list of statistic values from bootstrap samples.
plr, time = data_train.plr_si
plr_B_time_list = data_train.plr_si_B
```

**Parameters for `significance_test`**:
- `cov_index`: index of the covariate to be tested for significance.
- `B_seeds`: list of bootstrap seeds.
- `learning_rate`: learning rate for initial model training.
- `num_epoch`: number of epochs for initial model training.
- `num_epoch_`: number of epochs for model training during test statistic calculation.
- `num_epoch_B`: number of epochs for model training during bootstrap test statistic calculation.
- `learning_rate_B`: learning rate for model training during test statistic calculation.
- `Nets`: list of neural network classes for the main model (includes `cov_index`).
- `max_batch_size`: maximum batch size for training.
- `valid_rate`: proportion of data used for validation.
- `Nets_null`: list of neural network classes for the null model (excludes `cov_index`).
- `log`: file path for logging results.
- `num_model`: number of random initializations for model training during test statistic calculation.
- `num_model_B`: number of random initializations for model training during bootstrap test statistic calculation.
