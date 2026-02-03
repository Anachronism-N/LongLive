#!/bin/bash
# Example usage of I2V inference
# Usage: bash inference_i2v.sh <path_to_image> "<prompt>"

IMAGE=$1
PROMPT=$2

if [ -z "$IMAGE" ]; then
    echo "Usage: bash inference_i2v.sh <path_to_image> [\"<prompt>\"]"
    exit 1
fi

if [ -z "$PROMPT" ]; then
    # Default prompt if none provided
    PROMPT="high quality video"
fi

echo "Running I2V inference..."
echo "Image: $IMAGE"
echo "Prompt: $PROMPT"

# Use local GPU
export LOCAL_RANK=0
export WORLD_SIZE=1

torchrun \
  --nproc_per_node=1 \
  --master_port=29501 \
  inference_i2v.py \
  --config_path configs/longlive_inference_i2v.yaml \
  --image_path "$IMAGE" \
  --prompt "$PROMPT"
