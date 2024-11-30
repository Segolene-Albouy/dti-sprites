#!/bin/bash

module purge 
module load arch/a100
module load pytorch-gpu/py3/1.11.0
set -x

python -m src.sprites_trainer --multirun +dataset=clevr +model=clevr_mlp_proba +training=clevr ++training.cont=True # ++hydra.launcher.timeout_min=15 ++hydra.launcher.qos=qos_gpu_a100-dev 
