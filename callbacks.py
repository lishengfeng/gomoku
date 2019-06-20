from configs import FilepathConfig
import pickle


class CallbackList(object):
    """Container abstracting a list of callbacks.

    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_module_updated(self, batch):
        """Called when module is updated
        """
        for callback in self.callbacks:
            callback.on_module_updated(batch)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    """Abstract base class used to build new callbacks.

    on_module_updated:

    # Properties
        batch: The current batch
    """

    def __init__(self):
        self.filepath_config = FilepathConfig()
        self.model = None
        pass

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_module_updated(self, batch):
        pass


class ModelCheckpoint(Callback):
    """Save the model when new best model is found
    # Arguments
        filepath: string, path to save the model file.
    """

    def __init__(self):
        super().__init__()
        self.filepath = '{}.model'.format(self.filepath_config.filepath)

    def on_module_updated(self, batch):
        """ save model params to file """
        pickle.dump(self.model.get_weights(), open(self.filepath, 'wb'), protocol=2)
