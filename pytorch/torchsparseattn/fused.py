"""Fusedmax attention

Clusters neighboring attention weights into groups with equal weight.

A Regularized Framework for Sparse and Structured Neural Attention
Vlad Niculae, Mathieu Blondel
https://arxiv.org/abs/1705.07704
"""

from __future__ import division

import torch
from torch import nn
from torch import autograd as ta
import warnings
from lightning.impl.penalty import prox_tv1d

from .base import _BaseBatchProjection
from .sparsemax import SparsemaxFunction


def _inplace_fused_prox_jv_slow(y_hat, dout):
    """not efficient in python for long seqs, but template for a cython impl"""

    n_features = len(dout)

    for i in range(n_features + 1):
        if i in (0, n_features) or y_hat[i] != y_hat[i - 1]:
            if i > 0:
                dout[last_ix:i] = acc / n

            if i < n_features:
                last_ix = i
                acc = dout[i]
                n = 1
        else:
            acc += dout[i]
            n += 1
    return dout


try:
    from ._fused_jv import _inplace_fused_prox_jv
except ImportError:
    warnings.warn("Could not import cython implementation of fused backward "
                  "pass. Slow implementation used instead.")
    _inplace_fused_prox_jv = _inplace_fused_prox_jv_slow


def fused_prox_jv_slow(y_hat, dout):
    dout = dout.clone()
    _inplace_fused_prox_jv_slow(y_hat, dout)
    return dout


def fused_prox_jv_fast(y_hat, dout):
    dout = dout.clone()
    _inplace_fused_prox_jv(y_hat.detach().numpy(), dout.numpy())
    return dout


class FusedProxFunction(_BaseBatchProjection):

    def __init__(self, alpha=1):
        self.alpha = alpha

    def project(self, x):
        x_np = x.detach().numpy().copy()
        prox_tv1d(x_np, self.alpha)  # requires lightning/master for 32bit
        y_hat = torch.from_numpy(x_np)
        return y_hat

    def project_jv(self, dout, y_hat):
        dout = dout.clone()
        _inplace_fused_prox_jv(y_hat.detach().numpy(), dout.numpy())
        return dout


class Fusedmax(nn.Module):
    def __init__(self, alpha=1):
        self.alpha = alpha
        super(Fusedmax, self).__init__()

    def forward(self, x, lengths=None):
        fused_prox = FusedProxFunction(self.alpha)
        sparsemax = SparsemaxFunction()
        return sparsemax(fused_prox(x, lengths), lengths)


if __name__ == '__main__':
    from timeit import timeit
    torch.manual_seed(1)

    for dim in (5, 10, 50, 100, 500, 1000):

        x = torch.randn(dim)
        x_var = ta.Variable(x, requires_grad=True)
        y_hat = FusedProxFunction()(x_var).data
        dout = torch.arange(0, dim)
        print("dimension={}".format(dim))
        print("slow", timeit("fused_prox_jv_slow(y_hat, dout)",
                             globals=globals(),
                             number=10000))
        print("fast", timeit("fused_prox_jv_fast(y_hat, dout)",
                             globals=globals(),
                             number=10000))
