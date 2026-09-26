#!/usr/bin/env python3
"""Assert the sharp first-order direction law on a rational corank-two line.

The displayed diagonal example in the paper has a vanishing first-order
direction coefficient.  This fixed non-diagonal example verifies that the
O(delta) law can instead have a nonzero leading term.  It uses no random data.
"""

import numpy as np
from numpy.linalg import norm, svd
from scipy.optimize import brentq


def z_delta(matrix, delta):
    left, singular, right_t = svd(matrix, full_matrices=False)
    weights = singular / np.sqrt(singular * singular + delta * delta)
    return (left * weights) @ right_t


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

left, singular, right_t = svd(A_STAR, full_matrices=True)
rank = int(np.count_nonzero(singular > 1e-12))
compact_polar = left[:, :rank] @ right_t[:rank, :]
left_zero = left[:, rank:]
right_zero = right_t[rank:, :].T

a = float(np.sum(THETA * compact_polar))
c_block = left_zero.T @ THETA @ right_zero
p_c, sigma_c, q_c_t = svd(c_block, full_matrices=False)
beta = float(np.sum(sigma_c))

assert abs(a - 1.0 / 3.0) < 1e-14
assert abs(beta - 1.0) < 1e-14
assert abs(beta - 3.0 * abs(a)) < 1e-14


def eta_equation(eta):
    return float(
        np.sum(eta * sigma_c**2 / np.sqrt(1.0 + eta * eta * sigma_c**2))
        - abs(a)
    )


eta_r = brentq(eta_equation, 0.0, 10.0, xtol=1e-15, rtol=1e-14)
coefficients = eta_r * sigma_c / np.sqrt(1.0 + eta_r * eta_r * sigma_c**2)
k_r = -np.sign(a) * (p_c * coefficients) @ q_c_t
phi_r = compact_polar + left_zero @ k_r @ right_zero.T

assert abs(eta_r - 0.6448125266573337) < 2e-13
assert abs(a + float(np.sum(c_block * k_r))) < 2e-14
assert norm(phi_r, 2) <= 1.0 + 1e-13

deltas = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
rows = []
for delta in deltas:
    def stationarity(lam):
        return float(np.sum(THETA * z_delta(G + lam * THETA, delta)))

    left_value = stationarity(-2.0)
    right_value = stationarity(4.0)
    assert left_value < 0.0 < right_value
    lam_delta = brentq(
        stationarity, -2.0, 4.0, xtol=1e-15, rtol=1e-14
    )
    phi_delta = z_delta(G + lam_delta * THETA, delta)
    tangent_residual = abs(float(np.sum(THETA * phi_delta)))
    assert tangent_residual < 2e-12
    assert norm(phi_delta, 2) < 1.0
    multiplier_ratio = (lam_delta - LAMBDA_STAR) / delta
    direction_error = norm(phi_delta - phi_r, "fro")
    rows.append((delta, multiplier_ratio, direction_error))

ratios = np.array([row[1] for row in rows])
errors = np.array([row[2] for row in rows])
scaled_errors = errors / deltas
slopes = np.log(errors[1:] / errors[:-1]) / np.log(deltas[1:] / deltas[:-1])

assert abs(ratios[-1] + eta_r) < 2e-6
assert np.all(np.abs(ratios[1:] + eta_r) < np.abs(ratios[:-1] + eta_r))
assert np.all((0.98 < slopes) & (slopes < 1.02))
assert np.all((0.65 < scaled_errors) & (scaled_errors < 0.69))
assert scaled_errors[-1] > 0.6  # nonzero first-order direction coefficient

print("arbitrary-corank first-order direction check: PASS")
print(f"a={a:.12f}, beta={beta:.12f}, eta_R={eta_r:.12f}")
print("delta, (lambda_delta-lambda*)/delta, direction error, local slope")
for index, row in enumerate(rows):
    slope = "--" if index == 0 else f"{slopes[index - 1]:.6f}"
    print(f"  {row[0]:.0e}, {row[1]:+.10f}, {row[2]:.10e}, {slope}")
