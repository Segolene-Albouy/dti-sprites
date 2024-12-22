#!/bin/bash

module purge 
module load arch/a100
module load pytorch-gpu/py3/1.11.0
set -x

python -m src.sprites_trainer --multirun +dataset=clevr +model=clevr_mlp_proba +training=clevr hydra.job.name=gs-reg ++model.softmax=gumbel_softmax ++training.seed=7483,120,9847 ++model.lambda_empty_sprite=0.001,0.01 ++model.bin_weight=0.0001,0.001,0  # ++training.cont=True 
