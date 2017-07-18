import pytest
from sklearn.utils.testing import assert_allclose
import torch
from torch.autograd import gradcheck, Variable

from .fused import fused_prox_jv_slow, fused_prox_jv_la, fused_prox_jv_fast
from .fused import FusedProxFunction


def test_jv():

    torch.manual_seed(1)

    for _ in range(30):
        x = Variable(torch.randn(15))
        dout = torch.randn(15)
        y_hat = FusedProxFunction()(x).data

        din_1 = fused_prox_jv_la(y_hat, dout)
        din_2 = fused_prox_jv_slow(y_hat, dout)
        din_3 = fused_prox_jv_fast(y_hat, dout)
        assert_allclose(din_1.numpy(), din_2.numpy(), rtol=1e-5)
        assert_allclose(din_1.numpy(), din_3.numpy(), rtol=1e-5)


@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1, 1])
def test_finite_diff(alpha):
    torch.manual_seed(1)

    for _ in range(30):
        x = Variable(torch.randn(20), requires_grad=True)
        func = FusedProxFunction(alpha=alpha)
        assert gradcheck(func, (x,), eps=1e-4, atol=1e-3)
