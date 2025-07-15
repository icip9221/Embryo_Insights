class Registry:
    def __init__(self):
        self._registry = {}
        
    def register(self, name):
        if name in self._registry:
            print(f"{name} is already registered in {self._registry}")
            return
        
        def decorator(cls):
            self._registry[name] = cls
            return cls    
        
        return decorator
    
    def build(self, cfg: dict, *args, **kwargs):
        raise NotImplementedError
    
    def get(self, name: str, *args, **kwargs):
        return self._registry.get(name)(*args, **kwargs)
    
    def check(self, name: str):
        # Check if the class is existed in the registry
        return name in self._registry
    
class NormalRegistry:
    def __init__(self, register: Registry, module):
        super().__init__()
        self.register = register
        self.module = module
    
    def __call__(self, register_name):
        @self.register.register(register_name)
        class _Register(self.module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
        return _Register
        