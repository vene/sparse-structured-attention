import pytest
from numpy.testing import assert_allclose
import torch
from torch.autograd import gradcheck, Variable

from .fused import fused_prox_jv_slow, fused_prox_jv_fast
from .fused import FusedProxFunction


def _fused_prox_jacobian(y_hat, dout=None):
    """reference naive implementation: construct the jacobian"""
    dim = y_hat.size()[0]
    groups = torch.zeros(dim)
    J = torch.zeros(dim, dim)
    current_group = groups[0]

    for i in range(1, dim):
        if y_hat[i] == y_hat[i - 1]:
            groups[i] = groups[i - 1]
        else:
            current_group += 1
            groups[i] = current_group

    for i in range(dim):
        for j in range(dim):
            if groups[i] == groups[j]:
                n_fused = (groups == groups[i]).sum()
                J[i, j] = 1 / n_fused

    if dout is not None:
        return torch.mv(J, dout)
    else:
        return J


@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1, 1])
def test_jv(alpha):

    torch.manual_seed(1)

    for _ in range(30):
        x = Variable(torch.randn(15))
        dout = torch.randn(15)
        y_hat = FusedProxFunction(alpha=alpha)(x).data

        ref = _fused_prox_jacobian(y_hat, dout)
        din_slow = fused_prox_jv_slow(y_hat, dout)
        din_fast = fused_prox_jv_fast(y_hat, dout)
        assert_allclose(ref.numpy(), din_slow.numpy(), rtol=1e-5)
        assert_allclose(ref.numpy(), din_fast.numpy(), rtol=1e-5)


@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1, 1])
def test_finite_diff(alpha):
    torch.manual_seed(1)

    for _ in range(30):
        x = Variable(torch.randn(20), requires_grad=True)
        func = FusedProxFunction(alpha=alpha)
        assert gradcheck(func, (x,), eps=1e-4, atol=1e-3)
