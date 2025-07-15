#!/bin/bash

# Exit on any error
set -e

# Path to config file
CONFIG_PATH="config/Snake.yaml"
# CONFIG_PATH="config/GVFSnake.yaml"

# List of all image paths
IMAGES=(
  "data/microscope/1cell.jpg"
  "data/microscope/2cells.jpg"
  "data/microscope/3cells.jpg"
  "data/microscope/4cells.jpg"
  "data/microscope/5cells.jpg"
  "data/microscope/6cells.jpg"
  "data/microscope/7cells.jpg"
  "data/microscope/8cells.jpg"
)

# Optional: activate virtual environment
# source venv/bin/activate

# Run the script for each image
for IMAGE in "${IMAGES[@]}"; do
  echo "Processing $IMAGE"
  python main.py --pipeline_config "$CONFIG_PATH" --image "$IMAGE" --output "inference/microscope"
done

echo "✅ All images processed."
