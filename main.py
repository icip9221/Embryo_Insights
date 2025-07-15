import argparse
import os
import random

import cv2
import numpy as np
import yaml

from algorithms import *
from manager import BlastomereProcessHandlerManager
from registry import Blastomere_handler_register


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model with custom configurations.")
    parser.add_argument('--pipeline_config', 
                       type=str, 
                       help="Path to the pipeline configuration YAML file.")
    
    parser.add_argument('--image', 
                       default="sample/sample.jpg",
                       type=str, 
                       help="Path to sample imge.")
    
    parser.add_argument('--output', 
                       default="inference",
                       type=str, 
                       help="Path to sample imge.")
    
    
    return parser.parse_args()


def main(args: dict):
    input_filename = os.path.basename(args.get("image"))
    
    with open(args.get("pipeline_config"), "r") as f:
        pipeline_config = yaml.safe_load(f)
    
    manager = BlastomereProcessHandlerManager()

    builder_cls = pipeline_config.get("builder", None)
    if builder_cls is not None:
        if Blastomere_handler_register.check(builder_cls):
            builder = Blastomere_handler_register.get(builder_cls, cfg = pipeline_config['cfg'])
            manager.builder = builder
        else:
            raise ValueError(f"No builder class {builder_cls} found!")
    else:
        raise ValueError("No builder class specified in config file")

    manager.build_full_product()
    pipeline = manager.builder.product
    
    output_data = pipeline.process(data=args)

    annotated_image = np.copy(output_data["image"])
    final_mask = output_data.get("list_mask", [])

    for mask in final_mask:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        annotated_image[mask.astype(bool)] = color

    if output_data.get("uniform_eval") is not None:
        cv2.putText(annotated_image, f"NUV: {output_data.get('uniform_eval'):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
    name_without_ext = os.path.splitext(input_filename)[0]
    output_dir = os.path.join(args.get("output"), name_without_ext)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "annotated.jpg"), annotated_image)


    print("[DONE]")
    
if __name__ == "__main__":
    args = parse_args()
    main(vars(args))