
from tensorflow.keras import layers# pylint: disable=W0235

class Identity(layers.Layer):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x):
        """Return input"""
        return x
