import math

import torch
import numpy as np
import scipy.special
from numbers import Number


# ---------------------------------------------------------------------------
# GPU-native, log-space modified Bessel function of the first kind.
#
# Computes  log ive(v, z) = log( I_v(z) * exp(-z) )  for integer order v >= 0
# entirely on the GPU in pure PyTorch (so autograd flows through it and no
# scipy / CPU round-trip is needed).
#
# It uses the downward continued fraction for the ratio R_n = I_n(z)/I_{n-1}(z):
#       R_n = 1 / (2n/z + R_{n+1}),     R_{M+1} = 0
# and accumulates in log-space:
#       log ive(v, z) = log i0e(z) + sum_{n=1..v} log R_n
# Every R_n lies in (0, 1), so this never under/overflows -- it stays accurate
# in regimes (large v / small z, i.e. high latent dimension) where computing
# ive(v, z) directly underflows to 0 even in float64 and scipy returns -inf.
#
# The inner recurrence is a sequential loop of length M, so it is wrapped in
# torch.compile to fuse the iterations into efficient kernels (the eager loop
# launches one tiny kernel per step and is far slower than scipy).
# ---------------------------------------------------------------------------

_LOG_IVE_COMPILED = {}


def _log_ive_core(z, v, M):
    """Downward continued-fraction evaluation of log ive(v, z). float64 in/out."""
    R = torch.zeros_like(z)        # R_{M+1} = 0
    logsum = torch.zeros_like(z)   # accumulates sum_{n=1..v} log R_n
    for n in range(M, 0, -1):
        R = 1.0 / (2.0 * n / z + R)
        if n <= v:
            logsum = logsum + torch.log(R)
    return torch.log(torch.special.i0e(z)) + logsum


def _get_compiled(v, M):
    key = (v, M)
    fn = _LOG_IVE_COMPILED.get(key)
    if fn is None:
        fn = torch.compile(lambda z, _v=v, _M=M: _log_ive_core(z, _v, _M))
        _LOG_IVE_COMPILED[key] = fn
    return fn


def log_ive(v, z):
    """log( I_v(z) * exp(-z) ) for integer order ``v`` >= 0, GPU + autograd.

    ``z`` (the concentration) must be positive. Computed internally in float64
    and returned in ``z``'s dtype. The continued-fraction length ``M`` is chosen
    from the order and max(z) and bucketed so torch.compile only recompiles a
    handful of times across a run.
    """
    v = int(round(float(v)))
    z64 = z.double()
    zmax = float(z64.detach().abs().max())
    # M must exceed max(v, z) by a margin for the seeded recurrence to converge.
    M = v + int(math.ceil(zmax)) + 60
    M = ((M + 31) // 32) * 32  # bucket to limit torch.compile recompilations
    out = _get_compiled(v, M)(z64)
    return out.to(z.dtype)


class IveFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z):

        assert isinstance(v, Number), "v must be a scalar"

        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:  #  v > 0
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        #         else:
        #             print(v, type(v), np.isclose(v, 0))
        #             raise RuntimeError('v must be >= 0, it is {}'.format(v))

        return torch.Tensor(output).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return (
            None,
            grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z),
        )


class Ive(torch.nn.Module):
    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)


ive = IveFunction.apply


##########
# The below provided approximations were provided in the
# respective source papers, to improve the stability of
# the Bessel fractions.
# I_(v/2)(k) / I_(v/2 - 1)(k)

# source: https://arxiv.org/pdf/1606.02008.pdf
def ive_fraction_approx(v, z):
    # I_(v/2)(k) / I_(v/2 - 1)(k) >= z / (v-1 + ((v+1)^2 + z^2)^0.5
    return z / (v - 1 + torch.pow(torch.pow(v + 1, 2) + torch.pow(z, 2), 0.5))


# source: https://arxiv.org/pdf/1902.02603.pdf
def ive_fraction_approx2(v, z, eps=1e-20):
    def delta_a(a):
        lamb = v + (a - 1.0) / 2.0
        return (v - 0.5) + lamb / (
            2 * torch.sqrt((torch.pow(lamb, 2) + torch.pow(z, 2)).clamp(eps))
        )

    delta_0 = delta_a(0.0)
    delta_2 = delta_a(2.0)
    B_0 = z / (
        delta_0 + torch.sqrt((torch.pow(delta_0, 2) + torch.pow(z, 2))).clamp(eps)
    )
    B_2 = z / (
        delta_2 + torch.sqrt((torch.pow(delta_2, 2) + torch.pow(z, 2))).clamp(eps)
    )

    return (B_0 + B_2) / 2.0
