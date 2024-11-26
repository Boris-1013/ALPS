import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math


class TokenEmbedding(nn.Module):#实现了一个卷积神经网络（CNN）层，用于将输入的时间序列数据转换为嵌入表示
    def __init__(self, c_in, d_model, device=None):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)# 采用循环填充模式，使得卷积操作可以看作是循环的，这对时间序列建模可能更有利。
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
        #if device:
        #    self.to(device)

    def forward(self, x):
        # 获取卷积层权重所在的设备
        #device = self.tokenConv.weight.device
        #x=x.to(device)
        x = x.to(self.tokenConv.weight.dtype)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        #x.permute(0, 2, 1)：将输入张量的维度调整为 [batch_size, 特征数, 时间步数]，使得 Conv1d 可以在特征维度上进行卷积操作。
        #transpose(1, 2)：卷积完成后，再将张量的维度恢复为 [batch_size, 时间步数, 嵌入维度]，便于后续的模型层处理。
        return x


class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))#对输入的时间序列进行复制填充，确保在后续的补丁提取操作中，不会因为补丁（patch）长度或步长（stride）导致数据的边界溢出。

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        x = x.to(self.value_embedding.tokenConv.weight.dtype)
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars