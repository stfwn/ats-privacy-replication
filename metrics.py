import torch


class CrossEntropyLossWithFactor:
    def __init__(self, factor: float = 1.0):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.factor = factor

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        value = self.factor * self.loss_fn(x, y)
        return value
