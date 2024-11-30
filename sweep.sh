#!/bin/bash

module purge 
module load arch/a100
module load pytorch-gpu/py3/1.11.0
set -x

python -m src.sprites_trainer --multirun +dataset=dsprites_gray +model=dsprites_gray_mlp_proba +training=dsprites_gray ++model.softmax=softmax,gumbel_softmax ++model.lambda_empty_sprite=0.001,0.01 ++model.bin_weight=0.0001,0.001 ++training.cont=True # ++hydra.launcher.timeout_min=15 ++hydra.launcher.qos=qos_gpu_a100-dev 
