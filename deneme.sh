#!/bin/bash

module purge 
module load arch/a100
module load pytorch-gpu/py3/1.11.0
set -x

python -m src.sprites_trainer --multirun +dataset=tetrominoes +model=tetrominoes_mlp_proba +training=tetrominoes_proba ++training.seed=123,95,5069 ++model.bin_weight=0.0001 ++model.freq_weight=0,0.01,0.001  ++hydra.job.name=proba_regularization  

# ++hydra.launcher.timeout_min=15 ++hydra.launcher.qos=qos_gpu_a100-dev 
