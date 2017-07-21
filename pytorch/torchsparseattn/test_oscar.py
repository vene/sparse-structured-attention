import pytest
from sklearn.utils.testing import assert_allclose
import numpy as np
import torch
from torch.autograd import gradcheck, Variable

from .oscar import OscarProxFunction, oscar_prox_jv


def _oscar_prox_jacobian(y_star, dout=None):
    y_star = y_star.numpy()
    dim = y_star.shape[0]
    J = torch.zeros(dim, dim)

    _, inv, counts = np.unique(np.abs(y_star),
                               return_inverse=True,
                               return_counts=True)

    for i in range(dim):
        for j in range(dim):
            if (inv[i] == inv[j] and
                    y_star[i] != 0):
                J[i, j] = (np.sign(y_star[i]) * np.sign(y_star[j])
                           / counts[inv[i]])
    if dout is not None:
        return torch.mv(J, dout)
    else:
        return J


@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1, 1])
@pytest.mark.parametrize('beta', [0.001, 0.01, 0.1, 1])
def test_jv(alpha, beta):

    torch.manual_seed(1)

    for _ in range(30):
        x = Variable(torch.randn(15))
        dout = torch.randn(15)
        y_hat = OscarProxFunction(alpha=alpha, beta=beta)(x).data

        ref = _oscar_prox_jacobian(y_hat, dout)
        din = oscar_prox_jv(y_hat, dout)
        assert_allclose(ref.numpy(), din.numpy(), rtol=1e-5)


@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1, 1])
@pytest.mark.parametrize('beta', [0.001, 0.01, 0.1, 1])
def test_finite_diff(alpha, beta):
    torch.manual_seed(1)

    for _ in range(30):
        x = Variable(torch.randn(20), requires_grad=True)
        func = OscarProxFunction(alpha, beta=beta)
        assert gradcheck(func, (x,), eps=1e-5, atol=1e-3)
