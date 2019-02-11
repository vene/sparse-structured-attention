from torch import nn
from torch import autograd as ta

class _BaseBatchProjection(ta.Function):
    """Applies a sample-wise normalizing projection over a batch."""

    def forward(self, x, lengths=None):

        requires_squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            requires_squeeze = True

        n_samples, max_dim = x.size()

        has_lengths = True
        if lengths is None:
            has_lengths = False
            lengths = [max_dim] * n_samples

        y_star = x.new()
        y_star.resize_as_(x)
        y_star.zero_()

        for i in range(n_samples):
            y_star[i, :lengths[i]] = self.project(x[i, :lengths[i]])

        if requires_squeeze:
            y_star = y_star.squeeze()

        self.mark_non_differentiable(y_star)
        if has_lengths:
            self.mark_non_differentiable(lengths)
            self.save_for_backward(y_star, lengths)
        else:
            self.save_for_backward(y_star)

        return y_star

    def backward(self, dout):

        if not self.needs_input_grad[0]:
            return None

        if len(self.needs_input_grad) > 1 and self.needs_input_grad[1]:
            raise ValueError("Cannot differentiate {} w.r.t. the "
                             "sequence lengths".format(self.__name__))

        saved = self.saved_tensors
        if len(saved) == 2:
            y_star, lengths = saved
        else:
            y_star, = saved
            lengths = None

        requires_squeeze = False
        if y_star.dim() == 1:
            y_star = y_star.unsqueeze(0)
            dout = dout.unsqueeze(0)
            requires_squeeze = True

        n_samples, max_dim = y_star.size()
        din = dout.new()
        din.resize_as_(y_star)
        din.zero_()

        if lengths is None:
            lengths = [max_dim] * n_samples

        for i in range(n_samples):
            din[i, :lengths[i]] = self.project_jv(dout[i, :lengths[i]],
                                                  y_star[i, :lengths[i]])

        if requires_squeeze:
            din = din.squeeze()

        return din, None
