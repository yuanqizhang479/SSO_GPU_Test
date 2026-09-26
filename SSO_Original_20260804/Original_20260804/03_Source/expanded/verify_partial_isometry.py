#!/usr/bin/env python3
"""Assertions for certificate uniqueness and the two spectral completions."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm, svd
from scipy.optimize import brentq


def nuclear(a):
    return float(svd(a, compute_uv=False).sum())


def spectral(a):
    return float(svd(a, compute_uv=False)[0])


def compact_svd(a):
    u, s, vt = svd(a, full_matrices=False)
    keep = s > 1e-13 * max(1.0, s[0])
    return u[:, keep], s[keep], vt[keep, :].T


def hard_center(c, a):
    p, sigma, q = compact_svd(c)
    beta = float(sigma.sum())
    assert abs(a) <= beta + 1e-13
    if a == 0:
        return np.zeros_like(c), 0.0
    eta = brentq(
        lambda t: float(np.sum(sigma * np.minimum(t * sigma, 1.0))) - abs(a),
        0.0,
        1.0 / sigma[-1],
    )
    coeff = np.minimum(eta * sigma, 1.0)
    return -np.sign(a) * (p * coeff) @ q.T, eta


def smooth_center(c, a):
    p, sigma, q = compact_svd(c)
    beta = float(sigma.sum())
    assert abs(a) < beta
    if a == 0:
        return np.zeros_like(c), 0.0
    f = lambda t: float(
        np.sum(t * sigma**2 / np.sqrt(1.0 + t * t * sigma**2))
    ) - abs(a)
    hi = 1.0
    while f(hi) < 0:
        hi *= 2.0
    eta = brentq(f, 0.0, hi)
    coeff = eta * sigma / np.sqrt(1.0 + eta * eta * sigma**2)
    return -np.sign(a) * (p * coeff) @ q.T, eta


def check_partial_isometry_dichotomy():
    rng = np.random.default_rng(20260801)
    for _ in range(2000):
        c = rng.normal(size=tuple(rng.integers(1, 6, size=2)))
        lhs = norm(c, "fro") ** 2 / spectral(c)
        rhs = nuclear(c)
        assert lhs <= rhs + 2e-12 * max(1.0, rhs)

    examples = [
        np.outer(np.array([1.0, -2.0, 0.5]), np.array([0.4, 1.3])),
        1.7 * np.eye(4, 3),
    ]
    for c in examples:
        lhs = norm(c, "fro") ** 2 / spectral(c)
        assert abs(lhs - nuclear(c)) < 2e-12

    generic = np.diag([2.0, 1.0, 0.4])
    assert norm(generic, "fro") ** 2 / spectral(generic) < nuclear(generic)
    print("PASS  partial-isometry/no-clipping equivalence")


def check_completions():
    c = np.diag([2.0, 1.0, 0.4])
    beta = nuclear(c)
    a = 0.72 * beta

    kf, eta_f = hard_center(c, a)
    assert abs(float(np.sum(c * kf)) + a) < 2e-12
    assert spectral(kf) <= 1.0 + 2e-13
    # K_F is exactly the Euclidean projection of -eta_F C onto the ball.
    p, sigma, q = compact_svd(c)
    projected = -(p * np.minimum(eta_f * sigma, 1.0)) @ q.T
    assert norm(kf - projected, "fro") < 2e-13

    kr, eta_r = smooth_center(c, a)
    assert abs(float(np.sum(c * kr)) + a) < 2e-12
    assert spectral(kr) < 1.0
    coeff = svd(-kr, compute_uv=False)
    # The scalar KKT identity rho'(k_i)=eta_R sigma_i.
    assert norm(coeff / np.sqrt(1.0 - coeff**2) - eta_r * sigma) < 2e-11
    assert norm(kf - kr, "fro") > 1e-3

    # Rank-one data make the hard and smooth centers coincide.
    c1 = np.outer(np.array([1.0, -2.0]), np.array([0.3, 1.1, -0.4]))
    a1 = 0.37 * nuclear(c1)
    kf1, _ = hard_center(c1, a1)
    kr1, _ = smooth_center(c1, a1)
    assert norm(kf1 - kr1, "fro") < 2e-12
    print("PASS  hard and smooth spectral completion formulas")


def check_uniqueness_alternative():
    # Strict scalar slice: the equality fixes the sole entry.
    c_scalar = np.array([[2.0]])
    a_scalar = 0.7
    unique = np.array([[-a_scalar / 2.0]])
    assert spectral(unique) < 1.0
    assert abs(float(np.sum(c_scalar * unique)) + a_scalar) < 1e-15

    # Strict higher-dimensional slice: an orthogonal perturbation survives.
    c = np.diag([2.0, 1.0])
    a = 0.5
    k0 = -(a / nuclear(c)) * np.eye(2)
    h = np.array([[0.0, 1.0], [0.0, 0.0]])
    for sign in (-1.0, 1.0):
        k = k0 + sign * 0.1 * h
        assert spectral(k) < 1.0
        assert abs(float(np.sum(c * k)) + a) < 2e-15

    # Endpoint criterion: full rectangular rank means one complement vanishes.
    full_rect = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    deficient = np.diag([2.0, 0.0, 0.0])
    assert np.linalg.matrix_rank(full_rect) == min(full_rect.shape)
    assert np.linalg.matrix_rank(deficient) < min(deficient.shape)
    print("PASS  certificate-slice uniqueness alternatives")


def main():
    check_partial_isometry_dichotomy()
    check_completions()
    check_uniqueness_alternative()
    print("ALL CERTIFICATE-GEOMETRY ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
