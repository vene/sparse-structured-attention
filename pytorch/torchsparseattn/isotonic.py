"""
Isotonic Regression that preserves 32bit inputs.

backported from scikit-learn pull request
https://github.com/scikit-learn/scikit-learn/pull/9106"""

import numpy as np

from ._isotonic import _inplace_contiguous_isotonic_regression


def isotonic_regression(y, sample_weight=None, y_min=None, y_max=None,
                        increasing=True):
    """Solve the isotonic regression model::

        min sum w[i] (y[i] - y_[i]) ** 2

        subject to y_min = y_[1] <= y_[2] ... <= y_[n] = y_max

    where:
        - y[i] are inputs (real numbers)
        - y_[i] are fitted
        - w[i] are optional strictly positive weights (default to 1.0)

    Read more in the :ref:`User Guide <isotonic>`.

    Parameters
    ----------
    y : iterable of floating-point values
        The data.

    sample_weight : iterable of floating-point values, optional, default: None
        Weights on each point of the regression.
        If None, weight is set to 1 (equal weights).

    y_min : optional, default: None
        If not None, set the lowest value of the fit to y_min.

    y_max : optional, default: None
        If not None, set the highest value of the fit to y_max.

    increasing : boolean, optional, default: True
        Whether to compute ``y_`` is increasing (if set to True) or decreasing
        (if set to False)

    Returns
    -------
    y_ : list of floating-point values
        Isotonic fit of y.

    References
    ----------
    "Active set algorithms for isotonic regression; A unifying framework"
    by Michael J. Best and Nilotpal Chakravarti, section 3.
    """
    order = np.s_[:] if increasing else np.s_[::-1]
    # y = as_float_array(y)  # avoid sklearn dependency; we always pass arrays
    y = np.array(y[order], dtype=y.dtype)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=y.dtype)
    else:
        sample_weight = np.array(sample_weight[order], dtype=y.dtype)

    _inplace_contiguous_isotonic_regression(y, sample_weight)
    if y_min is not None or y_max is not None:
        # Older versions of np.clip don't accept None as a bound, so use np.inf
        if y_min is None:
            y_min = -np.inf
        if y_max is None:
            y_max = np.inf
        np.clip(y, y_min, y_max, y)
    return y[order]
