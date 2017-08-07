"""
From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label
Classification. André F. T. Martins, Ramón Fernandez Astudillo
In: Proc. of ICML 2016, https://arxiv.org/abs/1602.02068
"""

import numpy as np
import torch
from torch import nn
from .base import _BaseBatchProjection


def project_simplex_andre(a, radius=1.0):
    '''Project point a to the probability simplex.
    Returns the projected point x and the residual value.'''
    x0 = a.copy()
    d = len(x0)
    ind_sort = np.argsort(-x0)
    y0 = x0[ind_sort]
    ycum = np.cumsum(y0)
    val = 1.0 / np.arange(1, d + 1) * (ycum - radius)
    ind = np.nonzero(y0 > val)[0]
    rho = ind[-1]
    tau = val[rho]
    y = y0 - tau
    ind = np.nonzero(y < 0)
    y[ind] = 0
    x = x0.copy()
    x[ind_sort] = y
    return x, tau


def project_simplex_numpy(v):
    z = 1
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def project_simplex(v, z=1):
    v_sorted, _ = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - z
    ind = torch.arange(1, 1 + len(v))
    cond = v_sorted - cssv / ind > 0
    rho = ind.masked_select(cond)[-1]
    tau = cssv.masked_select(cond)[-1] / rho
    w = torch.clamp(v - tau, min=0)
    return w


def sparsemax_grad(dout, w_star):
    supp = w_star > 0
    masked = dout.masked_select(supp)
    masked -= masked.sum() / supp.sum()
    out = dout.new(dout.size()).zero_()
    out[supp] = masked
    return(out)


class SparsemaxFunction(_BaseBatchProjection):

    def project(self, x):
        return project_simplex(x)

    def project_jv(self, dout, y_star):
        return sparsemax_grad(dout, y_star)


class Sparsemax(nn.Module):

    def forward(self, x, lengths=None):
        sparsemax = SparsemaxFunction()
        return sparsemax(x, lengths)


def _bench():
    from timeit import timeit
    rng = np.random.RandomState(0)
    x = rng.randn(5).astype(np.float32)
    x_tn = torch.from_numpy(x)
    print("benchmarking")
    print(timeit("project_simplex_andre(x)", globals=globals()))
    print(timeit("project_simplex_numpy(x)", globals=globals()))
    print(timeit("project_simplex(x_tn)", globals=globals()))

if __name__ == '__main__':
    from torch.autograd import Variable

    n_samples = 5
    max_len = 10
    rng = np.random.RandomState(0)
    X = np.zeros((n_samples, max_len))
    mask = np.zeros((n_samples, max_len), dtype=np.int)
    for i in range(n_samples):
        # choose sequence length:
        d = rng.randint(0, max_len + 1)
        #  d = max_len
        X[i, :d] = rng.randn(d)
        mask[i, :d] = 1

    X_tn = Variable(torch.from_numpy(X).float(), requires_grad=True)
    mask_tn = Variable(torch.from_numpy(mask).byte())
    lengths = mask_tn.long().sum(1).detach()
    data = Variable(torch.randn(n_samples, 50, max_len))
    targets = Variable(torch.randn(n_samples))

    print(X_tn)
    sm = SparsemaxFunction()
    y = sm(X_tn, lengths)
    print(y)
    # compute a semblance of attention layer regression
    pred = torch.bmm(data, y.unsqueeze(2)).squeeze(2).mean(1)
    obj = ((pred - targets) ** 2).mean()
    obj.backward()

    print(X_tn.grad)
