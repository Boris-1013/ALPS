import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math


class TokenEmbedding(nn.Module):  
    """
    Implements a convolutional neural network (CNN) layer to convert input time series data into embedding representations.
    This layer uses a 1D convolution to encode feature representations from raw input.
    """
    def __init__(self, c_in, d_model, device=None):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model,
            kernel_size=3, 
            padding=padding, 
            padding_mode='circular',  # Circular padding ensures continuity, especially for cyclic patterns in time series data
            bias=False
        )
        # Initialize convolution weights using Kaiming normalization for stable training
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Forward pass for the TokenEmbedding layer.
        - Permutes input tensor to match Conv1D's expected input format.
        - Applies convolution and returns the transformed tensor.
        """
        x = x.to(self.tokenConv.weight.dtype)  # Match input data type with convolution weights
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # Permute, convolve, and revert to original format
        return x


class ReplicationPad1d(nn.Module):
    """
    Implements replication padding for 1D data.
    This is used to ensure continuity at the edges of the input sequence, 
    especially during patch extraction where strides may cause boundary issues.
    """
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for replication padding.
        - Replicates the last value of the input along the time dimension to match the required padding size.
        """
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)  # Concatenate original input with replicated values
        return output


class PatchEmbedding(nn.Module):
    """
    Implements a patch embedding mechanism for time series data.
    - Divides the input time series into smaller overlapping segments (patches).
    - Projects each patch into a higher-dimensional feature space using a convolutional layer.
    """
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len  # Length of each patch
        self.stride = stride  # Step size for patch extraction
        self.padding_patch_layer = ReplicationPad1d((0, stride))  # Ensures no boundary issues when extracting patches

        # Project each patch into a d-dimensional space using TokenEmbedding
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Residual dropout to prevent overfitting and improve generalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for PatchEmbedding:
        - Pads the input to ensure boundary alignment.
        - Extracts overlapping patches using unfolding.
        - Applies the TokenEmbedding layer to project patches into higher-dimensional embeddings.
        """
        x = x.to(self.value_embedding.tokenConv.weight.dtype)  # Match data type with embedding weights
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)  # Apply replication padding
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # Extract patches
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # Reshape for embedding layer
        x = self.value_embedding(x)  # Encode patches into embeddings
        return self.dropout(x), n_vars  # Apply dropout and return embeddings along with the number of features
