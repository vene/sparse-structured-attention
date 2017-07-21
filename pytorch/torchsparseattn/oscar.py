"""Oscarmax attention

Clusters attention weights into groups with equal weight, regardless of index.

A Regularized Framework for Sparse and Structured Neural Attention
Vlad Niculae, Mathieu Blondel
https://arxiv.org/abs/1705.07704
"""

import numpy as np
import torch
from torch import autograd as ta
from .isotonic import isotonic_regression


def oscar_prox_jv(y_hat, dout):
    y_hat = y_hat.numpy()
    din = dout.clone().zero_()
    dout = dout.numpy()
    din_np = din.numpy()

    sign = np.sign(y_hat)
    y_hat = np.abs(y_hat)

    uniq, inv, counts = np.unique(y_hat, return_inverse=True,
                                  return_counts=True)
    n_unique = len(uniq)
    tmp = np.zeros((n_unique,), dtype=y_hat.dtype)
    np.add.at(tmp, inv, dout * sign)
    tmp /= counts
    tmp.take(inv, mode='clip', out=din_np)
    din_np *= sign
    return din


def prox_owl(v, w):
    """Proximal operator of the OWL norm dot(w, reversed(sort(v)))

    Follows description and notation from:
    X. Zeng, M. Figueiredo,
    The ordered weighted L1 norm: Atomic formulation, dual norm,
    and projections.
    eprint http://arxiv.org/abs/1409.4271
    """

    # wlog operate on absolute values
    v_abs = np.abs(v)
    ix = np.argsort(v_abs)[::-1]
    v_abs = v_abs[ix]
    # project to K+ (monotone non-negative decreasing cone)
    v_abs = isotonic_regression(v_abs - w, y_min=0, increasing=False)

    # undo the sorting
    inv_ix = np.zeros_like(ix)
    inv_ix[ix] = np.arange(len(v))
    v_abs = v_abs[inv_ix]

    return np.sign(v) * v_abs


def _oscar_weights(alpha, beta, size):
    w = np.arange(size - 1, -1, -1, dtype=np.float32)
    w *= beta
    w += alpha
    return w


class OscarProxFunction(ta.Function):
    """Proximal operator of the OSCAR regularizer.

    ||w||_oscar = alpha ||w||_1 + beta * sum_i<j max { |w_i|, |w_j| }

    Implemented via the OWL norm with appropriate choice of weights, as
    described in:

    X. Zeng, M. Figueiredo,
    The ordered weighted L1 norm: Atomic formulation, dual norm,
    and projections.
    eprint http://arxiv.org/abs/1409.4271
    """

    def __init__(self, alpha=0, beta=1):
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        x_np = x.numpy().copy()
        weights = _oscar_weights(self.alpha, self.beta, x_np.shape[0])
        y_hat_np = prox_owl(x_np, weights)
        y_hat = torch.from_numpy(y_hat_np)
        self.save_for_backward(y_hat)
        return y_hat

    def backward(self, dout):
        if not self.needs_input_grad[0]:
            return None

        y_hat, = self.saved_tensors
        dout = oscar_prox_jv(y_hat, dout)
        return dout


if __name__ == '__main__':
    from timeit import timeit
    torch.manual_seed(1)

    for dim in (5, 10, 50, 100, 500, 1000):

        x = torch.randn(dim)
        x_var = ta.Variable(x, requires_grad=True)

        def _run_backward(x):
            y_hat = OscarProxFunction(beta=0.1)(x)
            val = y_hat.mean()
            val.backward()

        print("dimension={}".format(dim))
        print("la", timeit("_run_backward(x_var)",
                           globals=globals(),
                           number=10000))
