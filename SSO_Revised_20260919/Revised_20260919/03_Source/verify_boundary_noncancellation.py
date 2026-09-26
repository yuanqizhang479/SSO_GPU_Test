#!/usr/bin/env python3
"""100-digit check of the necessarily nonzero boundary direction coefficient.

This checks one explicit boundary model.  The general lower bound follows
from the orthogonality proof in the manuscript, not from this example.
"""

import json

import mpmath as mp


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def frobenius_pair(a, b):
    return mp.fsum(a[i, j] * b[i, j] for i in range(a.rows) for j in range(a.cols))


def spectral_data(lam):
    distance = mp.sqrt((lam - 1) ** 2 + 4)
    large = (lam + 1 + distance) / 2
    small = (lam + 1 - distance) / 2
    derivative_large = (1 + (lam - 1) / distance) / 2
    derivative_small = 1 - derivative_large
    return large, small, derivative_large, derivative_small


def smooth_output(lam, delta):
    large, small, _, _ = spectral_data(lam)
    a = mp.matrix([[lam, 1], [1, 1]])
    positive_projector = (a - small * mp.eye(2)) / (large - small)
    small_projector = mp.eye(2) - positive_projector
    return (large / mp.sqrt(large ** 2 + delta ** 2) * positive_projector
            + small / mp.sqrt(small ** 2 + delta ** 2) * small_projector)


def smooth_root(delta):
    def derivative(lam):
        large, small, dlarge, dsmall = spectral_data(lam)
        return (large * dlarge / mp.sqrt(large ** 2 + delta ** 2)
                + small * dsmall / mp.sqrt(small ** 2 + delta ** 2))

    lo, hi = mp.mpf(0), mp.mpf(1)
    require(derivative(lo) < 0 < derivative(hi), "boundary root bracket failed")
    for _ in range(360):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        value = derivative(mid)
        if value == 0:
            return mid
        if value < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    mp.mp.dps = 100
    half = mp.mpf(1) / 2
    dyad = mp.matrix([[half, -half], [-half, half]])
    p_derivative = mp.matrix([[mp.mpf(1) / 4, 0], [0, -mp.mpf(1) / 4]])
    d_derivative = -p_derivative
    coefficient = -p_derivative + d_derivative + dyad  # kappa/beta = 1.
    target = mp.matrix([[0, -half], [-half, 1]])
    require(coefficient == target, "analytic boundary coefficient mismatch")
    require(frobenius_pair(dyad, p_derivative) == 0, "positive/null orthogonality failed")
    require(frobenius_pair(dyad, d_derivative) == 0, "unit-dyad derivative failed")
    require(frobenius_pair(dyad, coefficient) == 1, "noncancellation pairing failed")
    require(frobenius_pair(coefficient, coefficient) == mp.mpf(3) / 2,
            "coefficient norm mismatch")

    exact_center = mp.matrix([[0, 1], [1, 0]])
    rows = []
    for delta in map(mp.mpf, ["1e-6", "1e-9"]):
        lam = smooth_root(delta)
        x = 1 - lam
        scaled_direction = (smooth_output(lam, delta) - exact_center) / x
        difference = scaled_direction - coefficient
        error = mp.sqrt(frobenius_pair(difference, difference))
        measured_pairing = frobenius_pair(dyad, scaled_direction)
        require(error < 2 * delta ** (mp.mpf(2) / 3),
                "scaled direction failed to approach nonzero coefficient")
        rows.append({"delta": mp.nstr(delta, 8),
                     "scaled_direction_error": mp.nstr(error, 30),
                     "dyad_pairing_of_scaled_direction": mp.nstr(measured_pairing, 30)})

    print(json.dumps({"arithmetic": "mpmath 100 digits, not interval arithmetic",
                      "kappa_over_beta": "1",
                      "coefficient": [["0", "-1/2"], ["-1/2", "1"]],
                      "coefficient_frobenius_norm": mp.nstr(mp.sqrt(mp.mpf(3) / 2), 30),
                      "results": rows}, indent=2))
    print("BOUNDARY NONCANCELLATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
