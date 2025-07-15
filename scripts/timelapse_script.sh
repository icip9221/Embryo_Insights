#!/bin/bash

# Exit on any error
set -e

# Path to config file
CONFIG_PATH="config/Snake.yaml"
# CONFIG_PATH="config/GVFSnake.yaml"

# List of all image paths
IMAGES=(
  "data/timelapse/1cell.png"
  "data/timelapse/2cells.png"
  "data/timelapse/3cells.png"
  "data/timelapse/4cells.png"
  "data/timelapse/5cells.png"
  "data/timelapse/6cells.jpg"
  "data/timelapse/7cells.png"
  "data/timelapse/8cells.png"
)

# Optional: activate virtual environment
# source venv/bin/activate

# Run the script for each image
for IMAGE in "${IMAGES[@]}"; do
  echo "Processing $IMAGE"
  python main.py --pipeline_config "$CONFIG_PATH" --image "$IMAGE" --output "inference/timelapse"
done

echo "✅ All images processed."
