import cv2
import numpy as np

from abstract import AbsAlgorithmProcessHandler, AbstractProcess
from registry import Blastomere_evaluation_process

from .utils import uniformity_evaluation


@Blastomere_evaluation_process.register("Uniformity")
class BlastomereCellUniformity(AbsAlgorithmProcessHandler):
    """
        This class computes the uniformity of blastomere cells based on their contour coordinates.

        The process involves:
        - Receiving a list of blastomere contours, where each contour is a list of (x, y) points.
        - Calculating the area of each contour using the Shoelace formula.
        - Evaluating the uniformity of the areas by computing the Normalized Unit Variance (NUV),
        which is the coefficient of variation (standard deviation divided by mean area).

        The resulting uniformity score reflects how similar the cell sizes are: 
        a lower value indicates more uniform cell areas.
    """
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug
        
    def _pre_process(self, data):
        return data
    
    def _main_process(self, data):
        contours = data.get("contours")
        uniform_eval = uniformity_evaluation(contours)
        return uniform_eval
    
    def _post_process(self, data):
        uniform_eval = data

        return uniform_eval
    
    def handle(self, data):
        uniform_eval = self.process(data)
        data.update(
            {
                "uniform_eval": uniform_eval,
            }
        )
        return super().handle(data)