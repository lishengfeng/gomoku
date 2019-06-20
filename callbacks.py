from configs import TrainConfig, BoardConfig


class Callback(object):
    """Abstract base class used to build new callbacks.

    on_module_updated:

    # Properties
        batch: The current batch
    """

    def __init__(self):
        self.train_config = TrainConfig()
        self.board_config = BoardConfig()
        pass

    def on_module_updated(self, batch):
        pass


class ModelCheckpoint(Callback):
    """Save the model when new best model is found

    """
