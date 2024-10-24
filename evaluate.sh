set -e
CURRENTDATE="0315"
run_tag="${CURRENTDATE}_$tag"
dataset="$(sed -n 2p configs/$config | cut -f2 -d ':' | cut -c2-)"
CUDA_VISIBLE_DEVICES=$cuda python -m ipdb src/trainer.py --tag $run_tag --config $config
