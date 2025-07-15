from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Handler(ABC):
    @abstractmethod
    def set_next(self, handler: Handler) -> Handler:
        pass
    
    @abstractmethod
    def handle(self, request: Any) -> Optional[str]:
        pass
    
    
class AbstractHandler(Handler):
    _next_handler: Handler = None
    
    def set_next(self, handler: Handler) -> Handler:
        self._next_handler = handler
        
        return handler
    
    @abstractmethod
    def handle(self, any: Any) -> Optional[str]:
        raise NotImplementedError
        


