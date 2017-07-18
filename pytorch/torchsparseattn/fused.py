"""Fusedmax attention

Clusters neighboring attention weights into groups with equal weight.

A Regularized Framework for Sparse and Structured Neural Attention
Vlad Niculae, Mathieu Blondel
https://arxiv.org/abs/1705.07704
"""

import numpy as np
import torch
from torch import autograd as ta

from lightning.impl.penalty import prox_tv1d


def fused_prox_jv_slow(y_hat, dout):
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


def fused_prox_jv_la(y_hat, dout):
    """vectorized implementation; slower on small sizes according to tests"""
    y_hat = y_hat.numpy()
    dim = len(y_hat)
    bounds = np.cumsum(np.ediff1d(y_hat, to_begin=0) != 0)
    L = np.zeros((dim, bounds[-1] + 1), dtype=y_hat.dtype)
    L[np.arange(dim), bounds] = 1
    L /= np.sqrt(L.sum(axis=0))
    L_t = torch.from_numpy(L)
    return torch.mv(L_t, torch.mv(L_t.t(), dout))


class FusedProxFunction(ta.Function):

    def __init__(self, alpha=1):
        self.alpha = alpha

    def forward(self, x):
        x_np = x.numpy().copy()
        prox_tv1d(x_np, self.alpha)  # requires lightning/master for 32bit
        y_hat = torch.from_numpy(x_np)
        self.save_for_backward(y_hat)
        return y_hat

    def backward(self, dout):

        if not self.needs_input_grad[0]:
            return None

        y_hat, = self.saved_tensors
        dout = fused_prox_jv_slow(y_hat, dout)

        return dout


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
        print("la", timeit("fused_prox_jv_la(y_hat, dout)",
                           globals=globals(),
                           number=10000))

