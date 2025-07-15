from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import onnxruntime
from torch import Tensor


class AbstractOnnxInference(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self):
        raise NotImplementedError
    
    @abstractmethod
    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()
        
    @abstractmethod
    def prepare_input(self):
        raise NotImplementedError
    
    @abstractmethod
    def inference(self, input_tensor: Tensor) -> Tensor:
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs

    @abstractmethod
    def process_output(self):
        raise NotImplementedError
    
    def draw_detections():
        ...

    @abstractmethod
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    @abstractmethod
    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]