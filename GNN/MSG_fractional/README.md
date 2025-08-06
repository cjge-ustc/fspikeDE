# Running MSG Model Using fspikeDE Method

This codebase implements experiments for **Graph Learning Tasks** described in the paper, covering two fundamental tasks:
- **Node Classification** 
- **Link Prediction**

You can train and evaluate the fspikeDE model on four datasets (Physics, CS, Amazon Photo, Amazon Computers) through parameter configurations.

---

## Execution Instructions

Run the following command to execute the fspikeDE model with IF neurons using Lorentz manifold on the Physics dataset:
```sh
python main.py --neuron "IF" --task "NC" --dataset "Physics" --manifold "lorentz"
```

---

### Detailed Parameter Description

More detailed parameter descriptions are as follows:
```sh
usage: main.py [-h] [--task {NC,LP}] [--dataset {computers,photo,KarateClub,CS,Physics}] [--root_path ROOT_PATH] [--eval_freq EVAL_FREQ] [--exp_iters EXP_ITERS] [--log_path LOG_PATH] [--epochs EPOCHS] [--lr LR] [--w_decay W_DECAY] [--use_MS] [--use_product]
               [--manifold MANIFOLD [MANIFOLD ...]] [--neuron {IF,LIF}] [--T T] [--n_layers N_LAYERS] [--embed_dim EMBED_DIM [EMBED_DIM ...]] [--step_size STEP_SIZE] [--v_threshold V_THRESHOLD] [--delta DELTA] [--tau TAU] [--dropout DROPOUT] [--lr_cls LR_CLS]
               [--w_decay_cls W_DECAY_CLS] [--epochs_cls EPOCHS_CLS] [--patience_cls PATIENCE_CLS] [--lr_lp LR_LP] [--w_decay_lp W_DECAY_LP] [--epochs_lp EPOCHS_LP] [--patience_lp PATIENCE_LP] [--t T] [--r R] [--use_gpu] [--gpu GPU] [--devices DEVICES]
               [--method {gl,predictor}]

Spiking Graph Neural Networks on Riemannian Manifold

optional arguments:
  -h, --help            show this help message and exit
  --task {NC,LP}
  --dataset {computers,photo,KarateClub,CS,Physics}
  --root_path ROOT_PATH
  --eval_freq EVAL_FREQ
  --exp_iters EXP_ITERS
  --log_path LOG_PATH
  --epochs EPOCHS
  --lr LR
  --w_decay W_DECAY
  --use_MS
  --use_product
  --manifold MANIFOLD [MANIFOLD ...]
                        Choose in combination [euclidean, lorentz, sphere]
  --neuron {IF,LIF}     Which neuron to use
  --T T                 latency of neuron
  --n_layers N_LAYERS
  --embed_dim EMBED_DIM [EMBED_DIM ...]
                        embedding dimension
  --step_size STEP_SIZE
                        step size for tangent vector
  --v_threshold V_THRESHOLD
                        threshold for neuron
  --delta DELTA         For LIF neuron
  --tau TAU
  --dropout DROPOUT
  --lr_cls LR_CLS
  --w_decay_cls W_DECAY_CLS
  --epochs_cls EPOCHS_CLS
  --patience_cls PATIENCE_CLS
  --lr_lp LR_LP
  --w_decay_lp W_DECAY_LP
  --epochs_lp EPOCHS_LP
  --patience_lp PATIENCE_LP
  --t T                 for Fermi-Dirac decoder
  --r R                 Fermi-Dirac decoder
  --use_gpu             use gpu
  --gpu GPU             gpu
  --devices DEVICES     device ids of multiple gpus
  --method {gl,predictor}
                        Which fde solver to use
```

> This part of code is borrowed from [Spiking Graph Neural Networks on Riemannian Manifolds](https://github.com/ZhenhHuang/MSG.git).
