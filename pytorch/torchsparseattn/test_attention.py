import pytest

import torch
from torch import nn
from torch.autograd import Variable

from . import Sparsemax, Fusedmax, Oscarmax


class AttentionRegressor(nn.Module):

    def __init__(self, projection, n_features=100):
        super(AttentionRegressor, self).__init__()
        self.projection = projection
        self.attn_template = nn.Parameter(torch.Tensor(n_features))
        self.attn_template.data.uniform_(-0.1, 0.1)

    def forward(self, X, lengths):

        # compute scores for each input word
        scores = torch.matmul(X, self.attn_template)
        weights = self.projection(scores, lengths)
        weighted_avg = torch.bmm(X.transpose(1, 2),
                                 weights.unsqueeze(-1)).squeeze(-1)
        pred = weighted_avg.sum(dim=1)  # very simple prediction rule
        return pred


@pytest.mark.parametrize('projection', [Sparsemax(),
                                        Fusedmax(0.1),
                                        Oscarmax(0.01)])
def test_attention(projection):
    n_samples = 20
    max_len = 10
    torch.manual_seed(1)
    n_features = 50

    X = torch.zeros(n_samples, max_len, n_features)

    # generate lengths in [1, max_len]
    lengths = 1 + (torch.rand(n_samples) * max_len).long()

    for i in range(n_samples):
        X[i, :lengths[i], :] = torch.randn(lengths[i], n_features)

    X = Variable(X)
    lengths = Variable(lengths)
    targets = Variable(torch.randn(n_samples))

    regr = AttentionRegressor(projection, n_features=n_features)
    loss_func = nn.MSELoss()
    optim = torch.optim.SGD(regr.parameters(), lr=0.0001)

    pred = regr(X, lengths)

    init_obj = loss_func(pred, targets)

    for it in range(50):
        optim.zero_grad()
        pred = regr(X, lengths)
        obj = loss_func(pred, targets)
        obj.backward()
        optim.step()

    final_obj = obj
    assert final_obj < init_obj
    assert regr.attn_template.grad.size() == (n_features,)
