import math
from typing import Union, Callable, cast

import torch
from torch import nn, Tensor
import torch.nn.functional as F 

ModuleType = Union[str, Callable[..., nn.Module]]


def get_model(config: dict) -> nn.Module:
    if config["model"] == "FTTransformer": # Used by MaskSDM
        return FTTransformer(
            feature_tokenizer=FeatureTokenizer(n_features=config["n_features"], d_token=config["d_hidden"]),
            transformer_encoder=TransformerEncoder(d_token=config["d_hidden"], n_blocks=config["n_blocks"], 
                                                   n_heads=config["n_heads"], dropout=config["dropout"]),
            d_hidden=config["d_hidden"],
            d_out=config["d_out"],
        )
    elif config["model"] == "MLP":
        return MLP(d_in=config["n_features"], d_hidden=config["d_hidden"], d_out=config["d_out"], 
                   n_layers=config["n_layers"], dropout=config["dropout"])
    elif config["model"] == "ResNet":
        return ResNet(d_in=config["n_features"], d_hidden=config["d_hidden"], d_out=config["d_out"], 
                      n_layers=config["n_layers"], dropout=config["dropout"])
    elif config["model"] == "linear" or config["model"] == "Maxent":
        return nn.Linear(config["n_features"], config["d_out"])
    else:
        raise ValueError(f"Unknown model: {config['model']}")


# The following code is a modified version of the code from https://github.com/yandex-research/rtdl/blob/main/rtdl/modules.py

class PredHead(nn.Module):

    def __init__(self, d_in: int, d_out: int, normalization: ModuleType, activation: ModuleType) -> None:
        super().__init__()
        self.normalization = normalization
        self.activation = activation
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: Tensor) -> Tensor:
        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.linear(x)
        return x


class MlpEncoder(nn.Module):
    
    class Block(nn.Module):

        def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))
    
    def __init__(self, d_in: int, d_out: int, n_layers: int, dropout: float) -> None:
        super(MlpEncoder, self).__init__()
        self.blocks = nn.Sequential(*[self.Block(d_in if i == 0 else d_out, d_out, dropout) for i in range(n_layers)])
        
    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)
    
    
class ResNetEncoder(nn.Module):
    
    class Block(nn.Module):

        def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
            super().__init__()
            self.normalization = nn.BatchNorm1d(d_in)
            self.linear1 = nn.Linear(d_in, d_out)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_out, d_out)

        def forward(self, x: Tensor) -> Tensor:
            return x + self.dropout(self.linear2(self.dropout(self.activation(self.linear1(self.normalization(x))))))
    
    def __init__(self, d_in: int, d_out: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.first_layer = nn.Linear(d_in, d_out)
        self.blocks = nn.Sequential(*[self.Block(d_out, d_out, dropout) for _ in range(n_layers)])
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        return self.blocks(x)
        
    
class TransformerEncoder(nn.Module):
    
    ffn_size_factor = 4/3
    
    class FFN(nn.Module):
        
        def __init__(self, d_token: int, d_hidden: int, dropout: float) -> None:
            super().__init__()
            self.linear1 = nn.Linear(d_token, d_hidden)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_hidden, d_token)
            
        def forward(self, x: Tensor) -> Tensor:
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x
    
    def __init__(self, d_token: int, n_blocks: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        
        assert d_token % n_heads == 0, 'd_token must be divisible by n_heads'
        d_hidden = int(d_token * self.ffn_size_factor)
        
        self.blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    'attention_normalization': nn.LayerNorm(d_token),
                    'attention': MultiheadAttention(d_token=d_token, n_heads=n_heads, dropout=dropout),
                    'attention_residual_dropout': nn.Dropout(dropout),
                    'ffn_normalization': nn.LayerNorm(d_token),
                    'ffn': self.FFN(d_token=d_token, d_hidden=d_hidden, dropout=dropout),
                    'ffn_residual_dropout': nn.Dropout(dropout),
                }
            )
            self.blocks.append(layer)
    
    def forward(self, x: Tensor) -> Tensor:
        
        for layer in self.blocks:
            layer = cast(nn.ModuleDict, layer)
            x_residual = layer['attention_normalization'](x)
            x_residual = layer['attention'](x_residual, x_residual)
            x_residual = layer['attention_residual_dropout'](x_residual)
            x = x + x_residual
            x_residual = layer['ffn_normalization'](x)
            x_residual = layer['ffn'](x_residual)
            x_residual = layer['ffn_residual_dropout'](x_residual)
            x = x + x_residual
            
        return x      
    
    
class MultiheadAttention(nn.Module):
    
    def __init__(self, d_token: int, n_heads: int, dropout: float) -> None:
        """
        Args:
            d_token: the token size. Must be a multiple of :code:`n_heads`.
            n_heads: the number of heads. If greater than 1, then the module will have
                an addition output layer (so called "mixing" layer).
            dropout: dropout rate for the attention map. The dropout is applied to
                *probabilities* and do not affect logits.
        """
        super().__init__()
        if n_heads > 1:
            assert d_token % n_heads == 0, 'd_token must be a multiple of n_heads'

        self.W_q = nn.Linear(d_token, d_token)
        self.W_k = nn.Linear(d_token, d_token)
        self.W_v = nn.Linear(d_token, d_token)
        self.W_out = nn.Linear(d_token, d_token) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x_q: query tokens
            x_kv: key-value tokens
        Returns:
            tokens
        """
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, "d_token must be divisible by n_heads"

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x
    

class FeatureTokenizer(nn.Module):
    """Feature Tokenizer module. It is used to embed features into tokens."""
    
    def __init__(self, n_features: int, d_token: int, periodic=True) -> None:
        super().__init__()
        self.periodic = periodic
        if self.periodic:
            self.periodic_embeddings = PeriodicEmbeddings(n_features=n_features, d_embedding=d_token)
        else:
            self.weight = nn.Parameter(Tensor(n_features, d_token))
            self.bias = nn.Parameter(Tensor(n_features, d_token))
            for parameter in [self.weight, self.bias]:
                if parameter is not None:
                    nn.init.uniform_(parameter, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
        
    def forward(self, x: Tensor) -> Tensor:
        x = torch.nan_to_num(x, nan=0.0) # For backward pass compatibility
        if self.periodic:
            return self.periodic_embeddings(x)
        else:
            x = self.weight[None] * x[..., None] + self.bias[None]
            return x
    
    
class _Periodic(nn.Module):

    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError(f'sigma must be positive, however: {sigma=}')

        super().__init__()
        self._sigma = sigma
        self.weight = nn.Parameter(torch.empty(n_features, k))
        self.reset_parameters()

    def reset_parameters(self):
        # NOTE[DIFF]
        # Here, extreme values (~0.3% probability) are explicitly avoided just in case.
        # In the paper, there was no protection from extreme values.
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        x = 2 * math.pi * self.weight * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html
class _NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings."""

    def __init__(self, n: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]
        x = (x[..., None, :] @ self.weight).squeeze(-2)
        x = x + self.bias
        return x

    
class PeriodicEmbeddings(nn.Module):

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 24,
        *,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01
    ) -> None:
        """
        Args:
            n_features: the number of features.
            d_embedding: the embedding size.
            n_frequencies: the number of frequencies for each feature.
                (denoted as "k" in Section 3.3 in the paper).
            frequency_init_scale: the initialization scale for the first linear layer
                (denoted as "sigma" in Section 3.3 in the paper).
                **This is an important hyperparameter**,
                see the documentation for details.
        """
        super().__init__()
        self.periodic = _Periodic(n_features, n_frequencies, frequency_init_scale)
        self.linear: Union[nn.Linear, _NLinear]
        self.linear = _NLinear(n_features, 2 * n_frequencies, d_embedding)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        x = self.periodic(x)
        x = self.linear(x)
        x = self.activation(x)
        return x
    
    
### Model classes    
    
class MLP(nn.Module):
    
    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.mlp_encoder = MlpEncoder(d_in=d_in, d_out=d_hidden, n_layers=n_layers, dropout=dropout)
        self.head = PredHead(d_in=d_hidden, d_out=d_out, normalization=None, activation=None)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.mlp_encoder(x))
    
    
class ResNet(nn.Module):
    
    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.resnet_encoder = ResNetEncoder(d_in=d_in, d_out=d_hidden, n_layers=n_layers, dropout=dropout)
        self.head = PredHead(d_in=d_hidden, d_out=d_out, normalization=nn.BatchNorm1d(d_hidden), activation=nn.ReLU())
        
    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.resnet_encoder(x))
    
    
class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer model."""
    
    def __init__(self, feature_tokenizer: FeatureTokenizer, transformer_encoder: TransformerEncoder, d_hidden: int, d_out: int, num_cls_tokens: int = 0) -> None:
        super().__init__()
        self.feature_tokenizer = feature_tokenizer
        self.satclip_projection = nn.Linear(256, d_hidden)
        self.transformer_encoder = transformer_encoder
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = PredHead(d_in=d_hidden, d_out=d_out, normalization=nn.LayerNorm(d_hidden), activation=nn.ReLU())
        self.mask_token = nn.Parameter(Tensor(d_hidden))
        for parameter in [self.mask_token]:
            nn.init.uniform_(parameter, -1 / math.sqrt(d_hidden), 1 / math.sqrt(d_hidden))
        self.num_cls_tokens = num_cls_tokens
        if num_cls_tokens > 0:
            self.num_cls_tokens = num_cls_tokens
            self.cls_tokens = nn.Parameter(Tensor(self.num_cls_tokens, d_hidden))
            for parameter in [self.cls_tokens]:
                nn.init.uniform_(parameter, -1 / math.sqrt(d_hidden), 1 / math.sqrt(d_hidden))
        
    def forward(self, x: Tensor, satclip_embeddings: Tensor = None, x_mask: Tensor = None, satclip_embeddings_mask: Tensor = None) -> Tensor:
        x = self.feature_tokenizer(x)
        if x_mask is not None:
            x = torch.where(x_mask.unsqueeze(-1).expand(x.shape) == 0, 
                            self.mask_token.unsqueeze(0).unsqueeze(0).expand(x.shape), 
                            x)
        if satclip_embeddings is not None:
            satclip_embeddings = self.satclip_projection(satclip_embeddings)
            if satclip_embeddings_mask is not None:
                satclip_embeddings = torch.where(satclip_embeddings_mask.unsqueeze(-1) == 0, 
                                                self.mask_token.unsqueeze(0).expand(satclip_embeddings.shape), 
                                                satclip_embeddings)
        if self.num_cls_tokens > 0:
            x = torch.cat([self.cls_tokens.repeat(len(x), 1, 1), x], dim=1)
        if satclip_embeddings is not None:
            x = torch.cat([x, satclip_embeddings.unsqueeze(1)], dim=1)
        x = self.transformer_encoder(x)
        if self.num_cls_tokens > 0:
            x = x[:, self.num_cls_tokens:, :]
        x = self.avgpool(x.transpose(1, 2)).squeeze(2)
        return self.head(x)