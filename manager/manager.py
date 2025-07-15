from abstract import AbsAlgorithmBuilder


class BlastomereProcessHandlerManager:
    def __init__(self) -> None:
        self._builder: AbsAlgorithmBuilder = None
    
    @property
    def builder(self) -> AbsAlgorithmBuilder:
        return self._builder
    
    @builder.setter
    def builder(self, builder: AbsAlgorithmBuilder) -> None:
        self._builder = builder
        
    def build_minimal_product(self) -> None:
        """
        The function `build_minimal_product` produces the main processor using a builder object.
        """
        self.builder.produce_main_processor()

    def build_full_product(self) -> None:
        """
        The function `build_full_product` calls methods to produce a pre-processor, main processor, and
        post-processor using a builder object.
        """
        self.builder.produce_pre_processor()
        self.builder.produce_main_processor()
        self.builder.produce_post_processor()
