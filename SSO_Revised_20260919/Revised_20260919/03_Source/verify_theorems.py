#!/usr/bin/env python3
"""Deterministic double-precision checks for the manuscript's key formulas.

These checks are numerical sanity tests, not computer-assisted proofs.
"""

from __future__ import annotations

# Scientific assertions must never disappear under python -O / PYTHONOPTIMIZE.
if not __debug__:
    raise RuntimeError("Verification requires assertions: rerun without -O / PYTHONOPTIMIZE.")

import math

import numpy as np
from numpy.linalg import norm, svd
from scipy.optimize import brentq


def nuclear(a: np.ndarray) -> float:
    return float(svd(a, compute_uv=False).sum())


def spectral(a: np.ndarray) -> float:
    return float(svd(a, compute_uv=False)[0])


def compact_polar(a: np.ndarray, rel_tol: float = 2e-12) -> np.ndarray:
    u, s, vt = svd(a, full_matrices=False)
    tol = rel_tol * max(1.0, float(s[0]))
    r = int(np.count_nonzero(s > tol))
    return u[:, :r] @ vt[:r, :]


def smooth_polar(a: np.ndarray, delta: float) -> np.ndarray:
    u, s, vt = svd(a, full_matrices=False)
    return (u * (s / np.sqrt(s * s + delta * delta))) @ vt


def inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b))


def h_compact(lam: float, g: np.ndarray, theta: np.ndarray) -> float:
    return inner(theta, compact_polar(g + lam * theta))


def h_smooth(lam: float, g: np.ndarray, theta: np.ndarray, delta: float) -> float:
    return inner(theta, smooth_polar(g + lam * theta, delta))


def assert_close(
    name: str,
    got: float,
    expected: float,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-10,
) -> None:
    err = abs(got - expected)
    limit = atol + rtol * abs(expected)
    if not np.isfinite(got) or err > limit:
        raise AssertionError(
            f"{name}: got {got:.12g}, expected {expected:.12g}, "
            f"error {err:.3e} > {limit:.3e}"
        )
    print(f"PASS  {name:<42} error={err:.3e}")


def test_t1_and_d1(rng: np.random.Generator) -> None:
    # Tall matrices are generically full column rank along the entire rank-one
    # line, so the compact scalar map is continuous and its root is ordinary.
    g = rng.normal(size=(5, 3))
    u = rng.normal(size=5)
    v = rng.normal(size=3)
    u /= norm(u)
    v /= norm(v)
    theta = np.outer(u, v)
    bound = 2.0 * nuclear(g) / nuclear(theta)
    lam_star = brentq(lambda x: h_compact(x, g, theta), -bound, bound)
    a_star = g + lam_star * theta
    phi = compact_polar(a_star)
    dual = nuclear(a_star)
    primal = inner(g, phi)
    assert_close("T1 primal-dual equality", primal, dual, rtol=2e-10)
    assert abs(inner(theta, phi)) < 2e-10
    assert abs(lam_star) <= bound * (1.0 + 1e-12)

    eps = 2e-5
    kappa = (
        h_compact(lam_star + eps, g, theta)
        - h_compact(lam_star - eps, g, theta)
    ) / (2.0 * eps)
    uu, ss, vvt = svd(a_star, full_matrices=False)
    b_star = -0.5 * inner(theta, (uu * (ss ** -2)) @ vvt)
    predicted = -b_star / kappa
    ratios = []
    for delta in (1e-2, 5e-3, 2e-3, 1e-3, 5e-4):
        root = brentq(
            lambda x: h_smooth(x, g, theta, delta), -bound, bound
        )
        ratios.append((root - lam_star) / (delta * delta))
    assert_close("T5-D1 leading constant", ratios[-1], predicted, rtol=4e-3)


def test_t2_interval_and_t3(rng: np.random.Generator) -> None:
    q = 4
    qu, _ = np.linalg.qr(rng.normal(size=(q, q)))
    qv, _ = np.linalg.qr(rng.normal(size=(q, q)))
    a_mat = qu @ np.diag([3.0, 1.2, 0.0, 0.0]) @ qv.T
    left = rng.normal(size=q)
    right = rng.normal(size=q)
    left /= norm(left)
    right /= norm(right)
    theta = np.outer(left, right)

    u, s, vt = svd(a_mat, full_matrices=True)
    r = 2
    ur, vr = u[:, :r], vt[:r, :].T
    u0, v0 = u[:, r:], vt[r:, :].T
    a = inner(theta, ur @ vr.T)
    c = u0.T @ theta @ v0
    beta = nuclear(c)
    eps = 2e-6
    right_deriv = (nuclear(a_mat + eps * theta) - nuclear(a_mat)) / eps
    left_deriv = (nuclear(a_mat) - nuclear(a_mat - eps * theta)) / eps
    assert_close("T2 right endpoint a+beta", right_deriv, a + beta, rtol=2e-5)
    assert_close("T2 left endpoint a-beta", left_deriv, a - beta, rtol=2e-5)

    # Directly check the rank-one canonical completion and random feasible
    # competitors in its affine hyperplane.
    kl, kr = 3, 4
    p = rng.normal(size=kl)
    qv0 = rng.normal(size=kr)
    p /= norm(p)
    qv0 /= norm(qv0)
    beta0, a0 = 0.8, 0.3
    c0 = beta0 * np.outer(p, qv0)
    kcan = -(a0 / beta0) * np.outer(p, qv0)
    assert_close("T3 canonical tangency", a0 + inner(c0, kcan), 0.0)
    accepted = 0
    for _ in range(5000):
        n = rng.normal(size=(kl, kr))
        n -= inner(c0, n) / inner(c0, c0) * c0
        t = rng.uniform(-0.35, 0.35)
        k = kcan + t * n / max(1.0, norm(n))
        if spectral(k) <= 1.0 + 1e-12:
            accepted += 1
            if norm(k) + 2e-12 < norm(kcan):
                raise AssertionError("T3 random feasible competitor beat K_can")
    if accepted < 1000:
        raise AssertionError("T3 test generated too few feasible competitors")
    print(f"PASS  {'T3 random feasible competitors':<42} accepted={accepted}")


def test_t4_and_beta_formula(rng: np.random.Generator) -> None:
    a, b, c = 0.7, 0.9, 1.4
    x_star = a * b / c
    amat = np.array([[x_star, a], [b, c]])
    jump = inner(np.array([[1.0, 0.0], [0.0, 0.0]]), compact_polar(amat))
    expected = a * b / math.sqrt((a * a + c * c) * (b * b + c * c))
    assert_close("T4 compact jump", jump, expected, rtol=2e-11)
    for x in (-0.4, 0.2, 0.9, 1.7):
        lhs = nuclear(np.array([[x, a], [b, c]])) ** 2
        rhs = (
            (x - c) ** 2 + (a + b) ** 2
            if x < x_star
            else (x + c) ** 2 + (a - b) ** 2
        )
        assert_close(f"T4 branch at x={x:g}", lhs, rhs, rtol=2e-12)

    max_rel = 0.0
    for n in range(2, 9):
        for _ in range(20):
            g = rng.normal(size=(n, n)) + 1.5 * np.eye(n)
            u = rng.normal(size=n)
            v = rng.normal(size=n)
            u /= norm(u)
            v /= norm(v)
            gu = np.linalg.solve(g, u)
            gtv = np.linalg.solve(g.T, v)
            alpha = float(v @ gu)
            if abs(alpha) < 1e-5:
                continue
            lam = -1.0 / alpha
            astar = g + lam * np.outer(u, v)
            ul, _, vtl = svd(astar, full_matrices=True)
            u0, v0 = ul[:, -1], vtl[-1, :]
            beta_svd = abs(u0 @ u) * abs(v @ v0)
            beta_closed = alpha * alpha / (norm(gu) * norm(gtv))
            rel = abs(beta_svd - beta_closed) / max(beta_closed, 1e-15)
            max_rel = max(max_rel, rel)
    if max_rel > 2e-8:
        raise AssertionError(f"SVD-free beta formula max relative error {max_rel}")
    print(f"PASS  {'SVD-free crossing half-width':<42} max_rel={max_rel:.3e}")


def test_d2_d3_and_old_remainder() -> None:
    theta = np.array([[1.0, 0.0], [0.0, 0.0]])

    g2 = np.array([[0.0, 0.5], [0.5, 1.0]])
    lam2 = 0.25
    beta2, a2 = 0.8, 0.2
    t0 = -a2 / (beta2 * math.sqrt(beta2 * beta2 - a2 * a2))
    vals = []
    for delta in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        root = brentq(lambda x: h_smooth(x, g2, theta, delta), -1.0, 1.5)
        vals.append((root - lam2) / delta)
    assert_close("T5-D2 strict-kink constant", vals[-1], t0, rtol=2e-4)

    g3 = np.array([[0.0, 1.0], [1.0, 1.0]])
    c0 = 2.0 ** (1.0 / 3.0)
    vals = []
    for delta in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        root = brentq(lambda x: h_smooth(x, g3, theta, delta), -0.5, 1.5)
        vals.append((1.0 - root) / (delta ** (2.0 / 3.0)))
    assert_close("T5-D3 boundary constant", vals[-1], c0, rtol=4e-3)

    # Counterexample to the old ungrouped o(|s|) remainder.
    slopes = []
    for s in (-1e-2, -3e-3, -1e-3, -3e-4, -1e-4):
        tau = (1.0 + s - math.sqrt(1.0 + s * s)) / 2.0
        tau_prime = (1.0 - s / math.sqrt(1.0 + s * s)) / 2.0
        q = math.copysign(1.0, tau) * tau_prime
        slopes.append((q + 0.5) / s)
    assert_close("D3 omitted linear term limit", slopes[-1], 0.5, rtol=2e-4)


def test_crossover() -> None:
    theta = np.array([[1.0, 0.0], [0.0, 0.0]])
    beta0, kappa0, zeta = 0.5, 0.5, 1.0
    y_expected = brentq(
        lambda y: kappa0 * y**3 + zeta * y**2 - 1.0 / (2.0 * beta0),
        1e-8,
        10.0,
    )
    transition = []
    far = []
    for delta in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6):
        eps = 2.0 * zeta * delta ** (2.0 / 3.0)
        c = math.sqrt(1.0 + eps)
        g = np.array([[0.0, 1.0], [1.0, c]])
        lam_star = 1.0 / c
        root = brentq(lambda x: h_smooth(x, g, theta, delta), -0.5, 1.5)
        transition.append((lam_star - root) / delta ** (2.0 / 3.0))

        eps_far = delta ** (1.0 / 3.0)
        cf = math.sqrt(1.0 + eps_far)
        gf = np.array([[0.0, 1.0], [1.0, cf]])
        lamf = 1.0 / cf
        rootf = brentq(lambda x: h_smooth(x, gf, theta, delta), -0.5, 1.5)
        den = math.sqrt((1.0 + cf * cf) ** 2)
        delta_gap = eps_far / den
        far.append((lamf - rootf) * math.sqrt(delta_gap) / delta)
    assert_close("T5-D4 transition constant", transition[-1], y_expected, rtol=2e-2)
    assert_close("T5-D4 strict-side constant", far[-1], 1.0, rtol=3e-2)


def test_t6(rng: np.random.Generator) -> None:
    m, n = 5, 4
    u = rng.normal(size=m)
    v = rng.normal(size=n)
    u /= norm(u)
    v /= norm(v)
    theta = np.outer(u, v)
    g = rng.normal(size=(m, n))
    z = 2.0 * rng.normal(size=(m, n))
    zbar = z / max(1.0, spectral(z))
    c = inner(theta, zbar)
    ztilde = (zbar - c * theta) / (1.0 + abs(c))
    assert abs(inner(theta, ztilde)) < 2e-14
    assert spectral(ztilde) <= 1.0 + 2e-14
    lam = 0.7
    amat = g + lam * theta
    gap = nuclear(amat) - inner(g, ztilde)
    rpol = nuclear(amat) - inner(amat, zbar)
    upper = rpol + abs(lam) * abs(c) + 2.0 * nuclear(g) * abs(c) / (1.0 + abs(c))
    if gap < -2e-12 or gap > upper + 2e-11:
        raise AssertionError("T6 gap coverage/decomposition failed")
    print(f"PASS  {'T6 feasible repair and gap bound':<42} gap={gap:.3e}")


def main() -> None:
    rng = np.random.default_rng(20260801)
    test_t1_and_d1(rng)
    test_t2_interval_and_t3(rng)
    test_t4_and_beta_formula(rng)
    test_d2_d3_and_old_remainder()
    test_crossover()
    test_t6(rng)
    print("\nAll deterministic numerical checks passed.")


if __name__ == "__main__":
    main()
