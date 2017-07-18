import pytest
from sklearn.utils.testing import assert_allclose
import torch
from torch.autograd import gradcheck, Variable

from .oscar import OscarProxFunction


@pytest.mark.parametrize('alpha', [0.001, 0.01, 0.1, 1])
@pytest.mark.parametrize('beta', [0.001, 0.01, 0.1, 1])
def test_finite_diff(alpha, beta):
    torch.manual_seed(1)

    for _ in range(30):
        x = Variable(torch.randn(20), requires_grad=True)
        func = OscarProxFunction(alpha, beta=beta)
        assert gradcheck(func, (x,), eps=1e-5, atol=1e-3)
