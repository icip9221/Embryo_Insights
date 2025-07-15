from abc import ABC, abstractmethod


class AbstractProcess(ABC):
    """
    Abstract base class for process with pre, main, post processing steps
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def _pre_process(self, data):
        pass
    
    @abstractmethod
    def _main_process(self, data):
        pass
    
    @abstractmethod
    def _post_process(self, data):
        pass
    
    def process(self, data):
        data = self._pre_process(data)
        data = self._main_process(data)
        data = self._post_process(data)
        return data
    
    def __call__(self, data):
        return self.process(data)
    
