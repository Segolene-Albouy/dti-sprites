#!/bin/bash

module purge 
module load arch/a100
module load pytorch-gpu/py3/1.11.0
set -x

python -m src.sprites_trainer --multirun +dataset=clevr6 +model=clevr6_mlp_proba +training=clevr6 ++model.softmax=gumbel_softmax,softmax ++training.seed=7483 ++hydra.job.name=longer ++training.cont=True 
#++model.lambda_empty_sprite=0.0001 ++model.bin_weight=0.001 # ++training.cont=True # ++hydra.launcher.timeout_min=15 ++hydra.launcher.qos=qos_gpu_a100-dev 
#python -m src.sprites_trainer --multirun +dataset=clevr6 +model=clevr6_mlp_proba +training=clevr6 ++model.softmax=gumbel_softmax ++training.seed=123 ++hydra.launcher.timeout_min=15 ++hydra.launcher.qos=qos_gpu_a100-dev ++hydra.job.name=deneme ++model.proba_type=marionette
