"""
Adapted from Assran, M. et al. Self-supervised learning from images with
a Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf.
Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/utils/tensors.py
(05.06.2024).
"""

import math

import numpy as np
import torch


def get_1d_sincos_pos_embed(embed_dim: int,
                            n_zero_pos: int,
                            n_sincos_pos: int,
                            ) -> np.ndarray:
    """
    Retrieve 1D sin cos positional embedding.

    Parameters
    -----------
    embed_dim:
        Output dimension of the positional embedding (for each
        position). Has to be divisible by 2.
    n_zero_pos:
        Number of positions to be embedded with 0s.
    n_sincos_pos:
        Number of positions to be embedded with sin cos positional
        embeddings.

    Returns
    -----------
    pos_embed:
        The positional embedding with shape (n_zero_pos+n_sincos_pos,
        embed_dim).
    """
    sincos_pos = np.arange(n_sincos_pos, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_pos(embed_dim, sincos_pos)
    if n_zero_pos > 0:
        pos_embed = np.concatenate(
            [np.zeros([n_zero_pos, embed_dim]), pos_embed],
            axis=0)

    return pos_embed


def repeat_interleave_batch(x: torch.Tensor, B: int, repeat: int
                            ) -> torch.Tensor:
    """
    Helper function to repeat tensors across batch dimension.

    Parameters
    -----------
    x:
        Tensor to repeat.
    B:
        Batch size.
    repeat:
        Number of times to repeat the tensor.   

    Returns
    -----------
    x:
        Tensor with repeated elements.
    """
    N = len(x) // B
    x = torch.cat(
            [torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
             for i in range(N)],
            dim=0)

    return x


def trunc_normal_(tensor: torch.Tensor,
                  mean: float = 0.,
                  std: float = 1.,
                  a: float = -2.,
                  b: float = 2.
                  ) -> torch.Tensor:
    """
    Helper function to initialize tensors with truncated normal
    distribution.

    Parameters
    -----------
    tensor:
        Tensor to initialize.
    mean:
        Mean of the normal distribution.
    std:
        Standard deviation of the normal distribution.
    a:
        Lower bound of the truncated normal distribution.
    b:
        Upper bound of the truncated normal distribution.

    Returns
    -----------
    tensor:
        Initialized tensor.
    """
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_1d_sincos_pos_embed_from_pos(embed_dim: int,
                                      pos: np.ndarray,
                                      ) -> np.ndarray:
    """
    Retrieve 1D sin cos positional embedding from an array of
    positions/sequence index.

    Parameters
    -----------
    embed_dim:
        Output dimension of the positional embedding (for each
        position). Has to be divisible by 2.
    pos:
        An array containing the positions to be embedded.
        
    Returns
    -----------
    pos_emb:
        The positional embedding with shape (len(pos), embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega # shape (EMBED_DIM/2,)

    pos = pos.reshape(-1) # shape (SEQ_LEN,)
    out = np.einsum('m,d->md', pos, omega) # shape (SEQ_LEN, EMBED_DIM/2);
                                           # outer product

    emb_sin = np.sin(out) # shape (SEQ_LEN, EMBED_DIM/2)
    emb_cos = np.cos(out) # shape (SEQ_LEN, EMBED_DIM/2)

    pos_emb = np.concatenate([emb_sin, emb_cos], axis=1) # shape (SEQ_LEN,
                                                         # EMBED_DIM)

    return pos_emb


def _no_grad_trunc_normal_(tensor: torch.Tensor,
                           mean: float,
                           std: float,
                           a: float,
                           b: float
                           ) -> torch.Tensor:
    """
    Cut & paste from PyTorch official master until it's in a few
    official releases - RW Method based on
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Parameters
    -----------
    tensor:
        Tensor to initialize.
    mean:
        Mean of the normal distribution.
    std:
        Standard deviation of the normal distribution.
    a:
        Lower bound of the truncated normal distribution.
    b:
        Upper bound of the truncated normal distribution.

    Returns
    -----------
    tensor:
        Initialized tensor.
    """
    def norm_cdf(x: float) -> float:
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution
        # and then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate
        # to [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get
        # truncated standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        
        return tensor