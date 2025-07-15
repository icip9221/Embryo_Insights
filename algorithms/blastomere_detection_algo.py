import cv2
import numpy as np

import adapter
from abstract import AbsAlgorithmProcessHandler, AbstractOnnxInference
from registry import Blastomere_evaluation_process


@Blastomere_evaluation_process.register("BlastomereCellLocalization")
class BlastomereCellLocalization(AbsAlgorithmProcessHandler):
    """
    This class handles the localization of blastomere cells using a pre-trained ONNX detection model.

    It performs the following steps:
    - Loads the image (if it's a file path).
    - Performs inference via an ONNX model to detect bounding boxes around blastomeres.
    - Returns the image, predicted bounding boxes, confidence scores, and class IDs.

    Required Config Parameters:
    - model: str, the model key defined in `adapter` for loading an ONNX model.
    - params: dict, parameters passed to the ONNX model constructor.

    Output (added to `data` dict):
    - "image": loaded image (np.ndarray)
    - "boxes": list of [x1, y1, x2, y2] bounding boxes
    - "scores": confidence scores for each detection
    - "class_ids": predicted class labels (usually unused for single-class problems)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.model: AbstractOnnxInference = getattr(adapter, config["model"])(**config["params"])

    def _pre_process(self, data: dict):
        image = data.get("image")
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image at path: {data['image']}")
        return image

    def _main_process(self, data):
        image = data
        boxes, scores, class_ids = self.model(image)
        return image, boxes, scores, class_ids

    def _post_process(self, data):
        return data

    def handle(self, data: dict):
        image, boxes, scores, class_ids = self.process(data)
        data.update(
            {
                "image": image,
                "boxes": boxes,
                "scores": scores,
                "class_ids": class_ids,
            }
        )
        return super().handle(data)

