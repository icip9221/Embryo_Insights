from abstract import (
    AbsAlgorithmBuilder,
    AbsAlgorithmProcessHandler,
    AbstractHandler,
    AbstractProcess,
)

from .registers import (
    Blastomere_evaluation_process,
    Blastomere_handler_register,
    Registry,
)


class BlastomereAlgorithm(AbstractProcess):
    def __init__(self):
        super().__init__()
        self.pre_processor: AbsAlgorithmProcessHandler = None
        self.main_processor: AbsAlgorithmProcessHandler = None
        self.post_processor: AbsAlgorithmProcessHandler = None

    def _pre_process(self, data):
        if self.pre_processor is not None:
            _data = self.pre_processor.handle(data)
        else:
            _data = data
        return _data

    def _main_process(self, data):
        if self.main_processor is not None:
            _data = self.main_processor.handle(data)
        else:
            _data = data
        return _data

    def _post_process(self, data):
        if self.post_processor is not None:
            _data = self.post_processor.handle(data)
        else:
            _data = data
        return _data

@Blastomere_handler_register.register("Blastomere")
class BlastomereAlgorithmBuilder(AbsAlgorithmBuilder):
    """
    The Builder for Object Measurement with Road Lane Detection
    """

    def __init__(self, cfg: dict) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        super().__init__(cfg)
        self.reset()

    def reset(self) -> BlastomereAlgorithm:
        """
        The function `reset` initializes the `_product` attribute with a new instance of
        `ObjMeasurementAlgorithm`.
        """
        self._product = BlastomereAlgorithm()

    @property
    def product(self) -> BlastomereAlgorithm:
        """
        The function returns the product of the measurements and then resets the object.

        Returns:
          The `product` variable is being returned.
        """
        product = self._product
        self.reset()
        return product

    def produce_pre_processor(self) -> None:
        """
        Create pre-processing pipeline
        """
        # empty pipeline
        ...

    def produce_main_processor(self) -> None:
        """
        Create the main processing pipeline for Lane Based Object Measurement.

        This function creates the main processing pipeline by iterating over the
        'mainprocessing' section of the algorithm_pipeline configuration. For each
        handler specified in the configuration, it checks if the handler is registered
        with the handler_register. If the handler is registered, it creates an instance
        of the handler and adds it to the list of handlers. If the handler is not
        registered, it raises a ValueError.

        After creating all the handlers, it creates a pipeline by setting the next
        handler for each handler in the list. Finally, it sets the main processor of
        the product to the first handler in the list.

        """
        # Create a list to store the handlers
        handlers = []

        # Iterate over the mainprocessing section of the configuration
        for handler_name, param in self._cfg["pipeline"].items():
            if Blastomere_evaluation_process.check(handler_name):
                handlers.append(Blastomere_evaluation_process.get(handler_name, **param)) if param is not None else handlers.append(Blastomere_evaluation_process.get(handler_name))
        # If there are handlers, create a pipeline
        if handlers:
            for i in range(1, len(handlers)):
                # Set the next handler for each handler in the list
                prev_handler: AbstractHandler = handlers[i - 1]
                cur_handler: AbstractHandler = handlers[i]
                prev_handler.set_next(cur_handler)

            # Set the main processor of the product to the first handler
            self._product.main_processor = handlers[0]

    def produce_post_processor(self) -> None:
        """
        Create pre-processing pipeline
        """
        # empty pipeline
        ...
