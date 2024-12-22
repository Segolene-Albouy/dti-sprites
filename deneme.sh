#!/bin/bash

module purge 
module load arch/a100
module load pytorch-gpu/py3/1.11.0
set -x

python -m src.sprites_trainer --multirun +dataset=tetrominoes +model=tetrominoes +training=tetrominoes ++training.seed=4321,872,958 ++hydra.job.name=dti-old
