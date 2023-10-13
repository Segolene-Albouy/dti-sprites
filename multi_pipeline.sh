#!/bin/bash
set -e
CURRENTDATE=`date +"%m%d"`
run_tag="${CURRENTDATE}_$tag"
dataset="$(sed -n 2p configs/$config | cut -f2 -d ':' | cut -c2-)"
for i in {0..9} 
do 
    seed=$(shuf -i 1-100000 -n 1)
    sed -i "s/seed:.*/seed: $seed/" configs/$config
    CUDA_VISIBLE_DEVICES=$cuda python src/sprites_trainer.py --tag ${run_tag}_$i --config $config
done

echo All runs are finished.
