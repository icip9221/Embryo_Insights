from abc import abstractmethod
from typing import Any

from .abs_handler import AbstractHandler
from .abs_process import AbstractProcess


class AbsAlgorithmProcessHandler(AbstractHandler, AbstractProcess):
    @abstractmethod
    def handle(self, data: Any):
        if self._next_handler:
            return self._next_handler.handle(data)
        
        return data
    
    @abstractmethod
    def _pre_process(self, data):
        raise NotImplementedError()
    
    @abstractmethod
    def _main_process(self, data):
        raise NotImplementedError()
    
    def _post_process(self, data):
        raise NotImplementedError()
    
    