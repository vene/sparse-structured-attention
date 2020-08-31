# Sparse and structured attention mechanisms
[![Build Status](https://travis-ci.org/vene/sparse-structured-attention.svg?branch=master)](https://travis-ci.org/vene/sparse-structured-attention)
[![PyPI version](https://badge.fury.io/py/torchsparseattn.svg)](https://badge.fury.io/py/torchsparseattn)

<p align="center"><img src="fusedmax.png" /></p>

--------------------------------------------------------------------------------

Efficient implementation of structured sparsity inducing
attention mechanisms: fusedmax, oscarmax and sparsemax.

**Note**: If you are just looking for sparsemax, I recommend the implementation in the [entmax](https://github.com/deep-spin/entmax).

Currently available for pytorch >= 0.4.1. (For older versions, use a previous
release of this package.) Requires python >= 2.7, cython, numpy, scipy.

Usage example:

```python

In [1]: import torch
In [2]: import torchsparseattn
In [3]: a = torch.tensor([1, 2.1, 1.9], dtype=torch.double)
In [4]: lengths = torch.tensor([3])
In [5]: fusedmax = torchsparseattn.Fusedmax(alpha=.1)
In [6]: fusedmax(a, lengths)
Out[6]: tensor([0.0000, 0.5000, 0.5000], dtype=torch.float64)
```

For details, check out our paper:

> Vlad Niculae and Mathieu Blondel
> A Regularized Framework for Sparse and Structured Neural Attention
> In: Proceedings of NIPS, 2017. 
> https://arxiv.org/abs/1705.07704 

See also:

> André F. T. Martins and Ramón Fernandez Astudillo
> From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification
> In: Proceedings of ICML, 2016
> https://arxiv.org/abs/1602.02068

> X. Zeng and M. Figueiredo,
> The ordered weighted L1 norm: Atomic formulation, dual norm, and projections.
> eprint http://arxiv.org/abs/1409.4271

