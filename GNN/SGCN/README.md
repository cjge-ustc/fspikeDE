# Running SGCN and DRSGNN Models Using SpikingJelly or SpikeDE

This codebase implements experiments for **Graph Learning Tasks** described in the paper, covering two major tasks:
- Spiking Graph Convolutional Networks (SGCN)
- Dynamic Reactive Spiking Graph Neural Network (DRSGNN)

Both models can be trained and evaluated on six datasets: Cora, Citeseer, Pubmed, Amazon Photo, Amazon Computers, and OGBN-Arxiv using either the SpikingJelly or SpikeDE framework.

---

## Execution Instructions

### Training from Scratch
To train models from scratch, use the command:
```sh
python train.py --dataset cora --model spikede --task DRSGCN \
    --batch_size 512 \
    --split 0.7 0.2 0.1 \
    --tau 2.0 \
    --tau_learnable \
    --threshold 1.0 \
    --integrator_indicator fdeint_adjoint \
    --integrator_method predictor-o \
    --positional_method random_walk \
    --positional_dim 32 \
    --beta 0.5 \
    --time_steps 100 \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --epochs 100 \
    --device cuda:0
```
This command trains the SpikeDE model on the Cora dataset.

### Loading Pretrained Models
To evaluate pretrained models, run:
```sh
python test.py --dataset cora --model spikede --task DRSGCN \
    --batch_size 512 \
    --split 0.7 0.2 0.1 \
    --tau 2.0 \
    --tau_learnable \
    --threshold 1.0 \
    --integrator_indicator fdeint_adjoint \
    --integrator_method predictor-o \
    --positional_method random_walk \
    --positional_dim 32 \
    --beta 0.5 \
    --time_steps 100 \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --epochs 100 \
    --device cuda:0
```
This loads and evaluates the pretrained model from `outputs/DRSGCN/cora/spikede_fde.pth`.

---

## Detailed Parameter Configuration

More detailed parameter configurations are as follows:
```sh
usage: train.py [-h] --dataset {cora,pubmed,citeseer,amazon_photo,amazon_computers,ogbn-arxiv} --task {SGCN,DRSGCN} --model {spikede,spikingjelly} [--split SPLIT [SPLIT ...]] [--batch_size BATCH_SIZE] [--positional_method {random_walk,laplace}] [--positional_dim POSITIONAL_DIM] [--tau TAU] [--tau_learnable]
                [--threshold THRESHOLD] [--integrator_indicator {odeint,fdeint,odeint_adjoint,fdeint_adjoint}] [--integrator_method {euler,predictor-f,predictor-o,trap-f,trap-o,gl-f,gl-o,predictor,implicitl1,gl,trap}] [--beta BETA] [--time_steps TIME_STEPS] [--dropout DROPOUT] [--learning_rate LEARNING_RATE]
                [--epochs EPOCHS] [--output_dir OUTPUT_DIR] [--device DEVICE] [--edge_keep_ratio EDGE_KEEP_RATIO] [--test_dropout TEST_DROPOUT]

SpikeDE experiment program.

optional arguments:
  -h, --help            show this help message and exit
  --dataset {cora,pubmed,citeseer,amazon_photo,amazon_computers,ogbn-arxiv}
                        The name of dataset.
  --task {SGCN,DRSGCN}  Task type.
  --model {spikede,spikingjelly}
                        The type of model.
  --split SPLIT [SPLIT ...]
                        The ratio decides how to split the dataset as train, eval and test.
  --batch_size BATCH_SIZE
                        Batch size.
  --positional_method {random_walk,laplace}
                        The method of positional encoding.
  --positional_dim POSITIONAL_DIM
                        The dim of positional code.
  --tau TAU             The initial tau.
  --tau_learnable       Whether tau is learnable.
  --threshold THRESHOLD
                        The initial threshold.
  --integrator_indicator {odeint,fdeint,odeint_adjoint,fdeint_adjoint}
                        The integrator indicator of SpikeDE model.
  --integrator_method {euler,predictor-f,predictor-o,trap-f,trap-o,gl-f,gl-o,predictor,implicitl1,gl,trap}
                        The integrator method of SpikeDE model. This parameter is up to integrator indicator: 'odeint' and 'odeint_adjoint' only support 'euler'; 'fdeint' only support 'predictor', 'implicitl1', 'gl' and 'trap'; 'fdeint_adjoint' only support 'predictor-f', 'predictor-o', 'trap-f', 'trap-o', 'gl-f' and
                        'gl-o'.
  --beta BETA           The beta param of SpikeDE model.
  --time_steps TIME_STEPS
                        The total time steps of spiking neural network.
  --dropout DROPOUT     The dropout p of models.
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        The learning rate of trainig.
  --epochs EPOCHS       Training epochs.
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        The path to experiment outputs.
  --device DEVICE       Which device to use.
  --edge_keep_ratio EDGE_KEEP_RATIO
                        How many edges to be kept.
  --test_dropout TEST_DROPOUT
                        The dropout p of graph features in test stage.
```

> When loading pretrained models, refer to the corresponding `txt` configuration file in the model directory to ensure consistent parameter settings, as mismatched configurations may lead to suboptimal performance.
