#!/usr/bin/env python3
"""Assertion-based checks of all three fixed-instance direction exponents."""

from __future__ import annotations

# Scientific assertions must never disappear under python -O / PYTHONOPTIMIZE.
if not __debug__:
    raise RuntimeError("Verification requires assertions: rerun without -O / PYTHONOPTIMIZE.")

import numpy as np
from numpy.linalg import norm, svd
from scipy.optimize import brentq


THETA = np.array([[1.0, 0.0], [0.0, 0.0]])


def smooth_polar(a, delta):
    u, s, vt = svd(a, full_matrices=False)
    return (u * (s / np.sqrt(s * s + delta * delta))) @ vt


def compact_polar(a):
    u, s, vt = svd(a, full_matrices=True)
    r = int(np.count_nonzero(s > 2e-12 * max(1.0, s[0])))
    return u[:, :r] @ vt[:r, :]


def exact_certificate(astar):
    u, s, vt = svd(astar, full_matrices=True)
    r = int(np.count_nonzero(s > 2e-12 * max(1.0, s[0])))
    partial = u[:, :r] @ vt[:r, :]
    if r == 2:
        return partial
    u0 = u[:, -1]
    v0 = vt[-1, :]
    alpha = float(u0 @ THETA @ v0)
    a = float(np.sum(THETA * partial))
    assert abs(alpha) > 1e-12
    return partial - (a / alpha) * np.outer(u0, v0)


def direction_error(g, lam_star, delta):
    root = brentq(
        lambda x: float(np.sum(THETA * smooth_polar(g + x * THETA, delta))),
        -2.0,
        2.0,
        xtol=5e-15,
        rtol=8.9e-16,
    )
    target = exact_certificate(g + lam_star * THETA)
    return norm(smooth_polar(g + root * THETA, delta) - target, "fro")


def regression_slope(deltas, errors):
    return float(np.polyfit(np.log(deltas), np.log(errors), 1)[0])


def check(name, g, lam_star, deltas, target_slope, tolerance):
    errors = np.array([direction_error(g, lam_star, d) for d in deltas])
    assert np.all(np.diff(errors) < 0), f"{name}: errors are not decreasing"
    slope = regression_slope(np.array(deltas), errors)
    assert abs(slope - target_slope) < tolerance, (
        f"{name}: slope {slope:.6f}, expected {target_slope:.6f}"
    )
    print(f"PASS  {name:<24} slope={slope:.6f}  errors={errors}")


def main():
    check(
        "regular direction",
        np.array([[0.0, 1.0], [1.0, 0.5]]),
        0.5,
        [1e-2, 3e-3, 1e-3, 3e-4],
        2.0,
        0.015,
    )
    check(
        "strict-kink direction",
        np.array([[0.0, 0.5], [0.5, 1.0]]),
        0.25,
        [1e-3, 3e-4, 1e-4, 3e-5],
        1.0,
        0.015,
    )
    check(
        "boundary direction",
        np.array([[0.0, 1.0], [1.0, 1.0]]),
        1.0,
        [1e-4, 3e-5, 1e-5, 3e-6],
        2.0 / 3.0,
        0.025,
    )
    print("ALL FIXED-INSTANCE DIRECTION-RATE ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
