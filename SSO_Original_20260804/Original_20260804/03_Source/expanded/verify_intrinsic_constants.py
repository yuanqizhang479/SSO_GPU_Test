#!/usr/bin/env python3
"""Assert the intrinsic strict-path constants and the sharp a=0 branch.

The first block checks Proposition 5.x on the fixed non-diagonal rational
corank-two line.  The second uses standard-library Decimal arithmetic to
check the cubic multiplier and quadratic direction coefficients when the
compact-polar equation is already correct at the singular minimizer.
"""

from decimal import Decimal, getcontext

import numpy as np
from numpy.linalg import norm, svd
from scipy.optimize import brentq


def z1(matrix):
    left, singular, right_t = svd(matrix, full_matrices=False)
    weights = singular / np.sqrt(1.0 + singular * singular)
    return (left * weights) @ right_t


def z_delta(matrix, delta):
    left, singular, right_t = svd(matrix, full_matrices=False)
    weights = singular / np.sqrt(singular * singular + delta * delta)
    return (left * weights) @ right_t


# Fixed non-diagonal rational corank-two line from the supplement.
A_STAR = np.diag([1.0, 3.0 / 5.0, 0.0, 0.0])
THETA = np.array(
    [
        [1.0 / 6.0, 0.0, 2.0 / 5.0, 0.0],
        [0.0, 1.0 / 6.0, 0.0, 3.0 / 10.0],
        [1.0 / 5.0, 0.0, 2.0 / 3.0, 0.0],
        [0.0, 1.0 / 10.0, 0.0, 1.0 / 3.0],
    ]
)
LAMBDA_STAR = 1.0
G = A_STAR - THETA
RANK = 2

left, _, right_t = svd(A_STAR, full_matrices=True)
compact = left[:, :RANK] @ right_t[:RANK, :]
left_zero = left[:, RANK:]
right_zero = right_t[RANK:, :].T
a = float(np.sum(THETA * compact))
c_block = left_zero.T @ THETA @ right_zero
sigma_c = svd(c_block, compute_uv=False)


def eta_equation(eta):
    return float(
        np.sum(eta * sigma_c**2 / np.sqrt(1.0 + eta * eta * sigma_c**2))
        - abs(a)
    )


eta_r = brentq(eta_equation, 0.0, 10.0, xtol=1e-15, rtol=1e-14)
t0 = -np.sign(a) * eta_r
d_r = float(np.sum(sigma_c**2 / (1.0 + t0 * t0 * sigma_c**2) ** 1.5))


def frozen_field(s, t):
    matrix = G + (LAMBDA_STAR + s) * THETA
    u_s, _, v_s_t = svd(matrix, full_matrices=True)
    p_big = u_s[:, :RANK] @ v_s_t[:RANK, :]
    u_zero = u_s[:, RANK:]
    v_zero = v_s_t[RANK:, :].T
    if s == 0.0:
        reduced = c_block
    else:
        reduced = (u_zero.T @ matrix @ v_zero) / s
    return p_big + u_zero @ z1(t * reduced) @ v_zero.T


step = 2e-3
d_w_s = (
    -frozen_field(2.0 * step, t0)
    + 8.0 * frozen_field(step, t0)
    - 8.0 * frozen_field(-step, t0)
    + frozen_field(-2.0 * step, t0)
) / (12.0 * step)
d_w_t = (
    -frozen_field(0.0, t0 + 2.0 * step)
    + 8.0 * frozen_field(0.0, t0 + step)
    - 8.0 * frozen_field(0.0, t0 - step)
    + frozen_field(0.0, t0 - 2.0 * step)
) / (12.0 * step)
pairing = float(np.sum(THETA * d_w_s))
t1 = -t0 * pairing / d_r
m_r = t0 * (d_w_s - (pairing / d_r) * d_w_t)
phi_r = frozen_field(0.0, t0)

assert abs(norm(m_r, "fro") - 0.6717753807) < 2e-9
assert abs(t1 - 0.7010530600) < 2e-10

delta = 1e-6


def stationarity(lam):
    return float(np.sum(THETA * z_delta(G + lam * THETA, delta)))


lam_delta = brentq(stationarity, -2.0, 4.0, xtol=1e-15, rtol=1e-14)
scaled_error = (z_delta(G + lam_delta * THETA, delta) - phi_r) / delta
cosine = float(np.sum(scaled_error * m_r) / (norm(scaled_error) * norm(m_r)))
assert abs(norm(scaled_error, "fro") - norm(m_r, "fro")) < 2e-6
assert cosine > 0.999999999

delta_t1 = 1e-5


def stationarity_t1(lam):
    return float(np.sum(THETA * z_delta(G + lam * THETA, delta_t1)))


lam_t1 = brentq(stationarity_t1, -2.0, 4.0, xtol=1e-15, rtol=1e-14)
observed_t1 = ((lam_t1 - LAMBDA_STAR) / delta_t1 - t0) / delta_t1
assert abs(observed_t1 - t1) < 8e-6


# Sharp compact-root branch: A*=diag(2,1,0), Theta=diag(1,-1,1).
getcontext().prec = 90
D = Decimal
ONE = D(1)
ZERO = D(0)


def bisect_increasing(fun, lo, hi, tol=D("1e-75")):
    f_lo, f_hi = fun(lo), fun(hi)
    assert f_lo < 0 < f_hi
    while hi - lo > tol:
        mid = (lo + hi) / D(2)
        if fun(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / D(2)


def compact_root_path(delta):
    def h(s):
        first = D(2) + s
        second = ONE - s
        return (
            first / (first * first + delta * delta).sqrt()
            - second / (second * second + delta * delta).sqrt()
            + s / (s * s + delta * delta).sqrt()
        )

    s = bisect_increasing(h, -delta, ZERO)
    phi = (
        (D(2) + s) / ((D(2) + s) ** 2 + delta * delta).sqrt(),
        (ONE - s) / ((ONE - s) ** 2 + delta * delta).sqrt(),
        s / (s * s + delta * delta).sqrt(),
    )
    tangent = phi[0] - phi[1] + phi[2]
    assert abs(tangent) < D("1e-65")
    error = sum((phi[index] - (ONE, ONE, ZERO)[index]) ** 2 for index in range(3)).sqrt()
    return s, error


predicted_multiplier = -D(3) / D(8)
predicted_direction = D(26).sqrt() / D(8)
compact_rows = []
for delta_decimal in map(D, ("1e-2", "1e-3", "1e-4", "1e-5")):
    displacement, direction_error = compact_root_path(delta_decimal)
    compact_rows.append(
        (
            delta_decimal,
            displacement / delta_decimal**3,
            direction_error / delta_decimal**2,
        )
    )

assert abs(compact_rows[-1][1] - predicted_multiplier) < D("2e-6")
assert abs(compact_rows[-1][2] - predicted_direction) < D("2e-6")
assert all(
    abs(compact_rows[index + 1][1] - predicted_multiplier)
    < abs(compact_rows[index][1] - predicted_multiplier)
    for index in range(len(compact_rows) - 1)
)
assert all(
    abs(compact_rows[index + 1][2] - predicted_direction)
    < abs(compact_rows[index][2] - predicted_direction)
    for index in range(len(compact_rows) - 1)
)

print("intrinsic first-order constants: PASS")
print(f"||M_R||_F={norm(m_r, 'fro'):.10f}, t1={t1:.8f}, cosine={cosine:.10f}")
print("sharp compact-root branch: PASS")
print(f"predicted multiplier coefficient = {predicted_multiplier}")
print(f"predicted direction coefficient = {predicted_direction}")
print("delta, displacement/delta^3, direction-error/delta^2")
for row in compact_rows:
    print("  " + ", ".join(str(value) for value in row))
