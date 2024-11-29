import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    A module for normalizing and denormalizing data, inspired by RevIN (Reversible Instance Normalization).
    It provides options for standard normalization, denormalization, and additional features like affine transformations.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):

        if mode == 'norm':
            self._get_statistics(x)  # Compute statistics for normalization
            x = self._normalize(x)  # Apply normalization
        elif mode == 'denorm':
            x = self._denormalize(x)  # Apply denormalization
