set -e
CURRENTDATE=`date +"%m%d"`
run_tag="${CURRENTDATE}_$tag"
dataset="$(sed -n 2p configs/$config | cut -f2 -d ':' | cut -c2-)"
seed=$(shuf -i 1-100000 -n 1)
sed -i "s/seed:.*/seed: $seed/" configs/$config
CUDA_VISIBLE_DEVICES=$cuda python -m src.kmeans_trainer --tag $run_tag --config $config
