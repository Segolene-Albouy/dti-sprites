#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR/venv/bin/activate"

# Function to show usage info
function show_usage {
    echo "Usage: $0 [options] [hydra_params]"
    echo "Options:"
    echo "  -t, --trainer TRAINER    Specify trainer: sprites (default) or kmeans"
    echo "  -d, --dataset DATASET    Specify dataset (e.g. dsprites_gray, tetro)"
    echo "  -m, --model MODEL        Specify model (e.g. dsprites_gray_mlp_proba)"
    echo "  -c, --config CONFIG      Specify training config"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -t sprites -d dsprites_gray -m dsprites_gray_mlp_proba ++model.softmax=gumbel_softmax"
    # bash run_trainer.sh -t sprites -d mnist -m mnist -c mnist
    exit 1
}

# Default values
TRAINER="sprites"
DATASET=""
MODEL=""
CONFIG=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--trainer)
            TRAINER="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            break
            ;;
    esac
done

# Validate trainer
if [[ "$TRAINER" != "sprites" && "$TRAINER" != "kmeans" ]]; then
    echo "Error: Trainer must be 'sprites' or 'kmeans'"
    show_usage
fi

# Validate dataset
if [[ -z "$DATASET" ]]; then
    echo "Error: Dataset must be specified with -d or --dataset"
    show_usage
fi

# Build command
# CMD="python -m src.${TRAINER}_trainer --multirun"
CMD="python -m src.${TRAINER}_trainer"

# Add dataset configuration
CMD="$CMD +dataset=$DATASET"

# Add model if specified
if [[ ! -z "$MODEL" ]]; then
    CMD="$CMD +model=$MODEL"
fi

# Add training config if specified
if [[ ! -z "$CONFIG" ]]; then
    CMD="$CMD +training=$CONFIG"
fi

# if additional Hydra parameters
if [[ $# -gt 0 ]]; then
    CMD="$CMD $@"
fi

# Execute command
echo "Running: $CMD"
eval $CMD
