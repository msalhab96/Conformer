import math
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from functools import lru_cache


class ConformerBlock(nn.Module):
    pass


class MHSA(nn.Module):
    def __init__(
            self,
            enc_dim: int,
            dk: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__()
        assert enc_dim % dk == 0, 'enc_dim is not divisible by dk'
        self.fc_key = nn.Linear(
            in_features=enc_dim,
            out_features=enc_dim,
        )
        self.fc_query = nn.Linear(
            in_features=enc_dim,
            out_features=enc_dim,
        )
        self.fc_value = nn.Linear(
            in_features=enc_dim,
            out_features=enc_dim,
        )
        self.lnorm = nn.LayerNorm(enc_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.enc_dim = enc_dim
        self.dk = dk
        self.sqrt_dk = math.sqrt(dk)
        self.h = enc_dim // dk
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def _key_query_matmul(self, Q: Tensor, K: Tensor) -> Tensor:
        """Performs the Matmul operation in
        scaled Dot-Product Attention

        Args:
            Q (Tensor): The Query tensor of shape [B, M, dk, h]
            K (Tensor): The Key tensor of shape [B, M, dk, h]

        Returns:
            Tensor: The result of matmul operation of shape
            [B, M, dk, dk]
        """
        return torch.matmul(Q, K.permute(0, 1, 3, 2))

    def _get_scaled_att(
            self,
            Q: Tensor,
            K: Tensor
            ) -> Tensor:
        """Calculates the scaled attention map
        by calculating softmax(matmul(Q, K.T)/sqrt(dk))

        Args:
            Q (Tensor): The Query tensor of shape [B, M, dk, h]
            K (Tensor): The Key tensor of shape [B, M, dk, h]

        Returns:
            Tensor: The scaled attention map of shape
            [B, M, dk, dk]
        """
        result = self._key_query_matmul(Q, K)
        result /= self.sqrt_dk
        return self.softmax(result)

    def perform_self_att(
            self,
            Q: Tensor,
            K: Tensor,
            V: Tensor
            ) -> Tensor:
        """Perform multi head scaled attention
        by calculating softmax(matmul(Q, K.T)/sqrt(dk)).V

        Args:
            Q (Tensor): The Query tensor of shape [B, M, dk, h]
            K (Tensor): The Key tensor of shape [B, M, dk, h]
            V (Tensor): The Value tensor of shape [B, M, dk, h]

        Returns:
            Tensor: The scaled attention map of shape
            [B, M, dk * h]
        """
        (b, m, *_) = Q.shape
        att = self._get_scaled_att(Q, K)
        result = torch.matmul(att, V)
        return result.view(b, m, -1)

    @lru_cache(maxsize=2)
    def get_positionals(self, max_length: int) -> Tensor:
        """Create Positionals tensor to be added to the input

        Args:
            max_length (int): The maximum length

        Returns:
            Tensor: Positional tensor
        """
        result = torch.zeros(max_length, self.enc_dim)
        pos = torch.arange(0, self.enc_dim).repeat(max_length, 1)
        i = torch.arange(0, max_length).repeat(self.enc_dim, 1).T
        result[:, 1::2] = torch.sin(
            pos[:, 1::2] / (10000 ** ((2 * i[:, 1::2]) / self.dk))
            )
        result[:, 0::2] = torch.cos(
            pos[:, 0::2] / (10000 ** ((2 * i[:, 0::2]) / self.dk))
            )
        return result

    def _reshape(self, *args) -> List[Tensor]:
        """Reshabes all the given list of tensor
        from [B, M, N] to [B, M, dk, h]

        Returns:
            List[Tensor]: list of all reshaped tensors
        """
        return [
            item.view(-1, item.shape[1], self.dk, self.h)
            for item in args
        ]

    def forward(self, inp: Tensor) -> Tensor:
        """Passes the input into multi-head attention

        Args:
            inp (Tensor): The input tensor

        Returns:
            Tensor: The result after adding it to positionals
            and passing it through multi-head self-attention
        """
        out = self.lnorm(inp)
        K = self.fc_key(out)
        Q = self.fc_query(out)
        V = self.fc_value(out)
        max_length = inp.shape[1]
        positionals = self.get_positionals(max_length).to(self.device)
        out += positionals
        (Q, K, V) = self._reshape(Q, K, V)
        out = self.perform_self_att(Q, K, V)
        out = self.dropout(out)
        return inp + out


class ConvModule(nn.Module):
    """Implements the convolution module
    where it contains the following layers
    1. Layernorm
    2. Pointwise Conv
    3. Gate Linear unit
    4. 1D Depthwise conv
    5. BatchNorm
    6. Swish Activation
    7. Pointwise Conv
    8. Dropout

    Args:
        enc_dim (int): The encoder dimensionality.
        scaling_factor (int): The scaling factor of the conv layer.
        kernel_size (int): The convolution kernel size.
        p_dropout (float): The dropout probability.
    """
    def __init__(
            self,
            enc_dim: int,
            scaling_factor: int,
            kernel_size: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.lnorm = nn.LayerNorm(enc_dim)
        n_scaled_channels = enc_dim * scaling_factor
        assert (kernel_size - 1) % 2 == 0, 'kernel_size - 1 \
            must be divisable by 2 -odd'
        padding_size = (kernel_size - 1) // 2
        self.pwise_conv1 = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=n_scaled_channels,
            kernel_size=1
        )
        self.glu = nn.GLU(dim=1)
        self.dwise_conv = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=enc_dim,
            kernel_size=kernel_size,
            padding=padding_size,
            groups=enc_dim
        )
        self.bnorm = nn.BatchNorm1d(enc_dim)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(p_dropout)
        self.pwise_conv2 = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=enc_dim,
            kernel_size=1
        )

    def forward(self, inp: Tensor) -> Tensor:
        out = self.lnorm(inp)
        out = out.permute(0, 2, 1)
        out = self.pwise_conv1(out)
        out = self.glu(out)
        out = self.dwise_conv(out)
        out = self.bnorm(out)
        out = self.swish(out)
        out = self.pwise_conv2(out)
        out = self.dropout(out)
        return out + inp


class FeedForwardModule(nn.Module):
    """Implements the feed forward module in the conformer block
    where the module consists of the below
    1. Layer Norm
    2. Linear Layer
    3. Swish Activation
    4. Dropout
    5. Linear Layer
    6. Dropout

    Args:
        enc_dim (int): The encoder dimensionality
        scaling_factor (int): The scaling factor of the linear layer
        p_dropout (float): The dropout probability
        residual_scaler (float, optional): The residual scaling.
        Defaults to 0.5.
    """
    def __init__(
            self,
            enc_dim: int,
            scaling_factor: int,
            p_dropout: float,
            residual_scaler=0.5
            ) -> None:
        super().__init__()
        self.residual_scaler = residual_scaler
        scaled_dim = scaling_factor * enc_dim
        self.lnorm = nn.LayerNorm(enc_dim)
        self.fc1 = nn.Linear(
            in_features=enc_dim,
            out_features=scaled_dim
        )
        self.fc2 = nn.Linear(
            in_features=scaled_dim,
            out_features=enc_dim
        )
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, inp: Tensor, x=4) -> Tensor:
        """Passes the given inp through the feed forward
        module

        Args:
            inp (Tensor): the input to the feed forward module
            with shape [B, M, N] where B is the batch size, M
            is the maximum length, and N is the encoder dim
        Returns:
            Tensor: The result of the forward module
        """
        out = self.lnorm(inp)
        out = self.fc1(out)
        out = self.swish(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.residual_scaler * inp + out


class Model(nn.Module):
    pass
