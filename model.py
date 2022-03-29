import math
from turtle import forward
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


class ConvModule(nn.Module):
    pass


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
