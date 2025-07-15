from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .abs_process import AbstractProcess


class AbsAlgorithmBuilder(ABC):
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        
    @property
    @abstractmethod
    def product(self) -> AbstractProcess:
        "Get the product"
        pass

    @abstractmethod
    def produce_pre_processor(self) -> None:
        "build pre-processor"
        ...

    @abstractmethod
    def produce_main_processor(self) -> None:
        "build main processor"
        ...

    @abstractmethod
    def produce_post_processor(self) -> None:
        "build post-processor"
        ...