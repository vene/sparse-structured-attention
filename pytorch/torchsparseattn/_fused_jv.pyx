cimport cython
from cython cimport floating


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _inplace_fused_prox_jv(floating[::1] y_hat, floating[::1] dout):
    cdef Py_ssize_t n_features = dout.shape[0]
    cdef Py_ssize_t i, last_ix
    cdef unsigned int n
    cdef floating acc
    for i in range(n_features + 1):
        if i in (0, n_features) or y_hat[i] != y_hat[i - 1]:
            if i > 0:
                dout[last_ix:i] = acc / n

            if i < n_features:
                last_ix = i
                acc = dout[i]
                n = 1

        else:
            acc += dout[i]
            n += 1
    return dout



