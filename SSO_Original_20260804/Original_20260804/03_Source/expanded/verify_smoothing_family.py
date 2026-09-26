#!/usr/bin/env python3
"""Deterministic checks for the smoothing-family and cancellation results.

The float64 block checks the two distinguished centers, polynomial-tail
boundary exponents, finite-saturation constants, and Huber multiplier
nonuniqueness.  The mpmath block verifies the quintic/quartic compact-root
coefficients at 100 digits.
"""

import math

import mpmath as mp
import numpy as np


def inner(x, y):
    return float(np.sum(x * y))


def z_spectral(x, delta, coefficient):
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return (u * coefficient(s / delta)) @ vt


def z_ph(x, delta):
    return z_spectral(x, delta, lambda y: y / np.sqrt(1.0 + y * y))


def z_huber(x, delta, threshold=1.0):
    return z_spectral(x, delta, lambda y: np.minimum(y / threshold, 1.0))


def z_power_tail(x, delta, p):
    return z_spectral(x, delta, lambda y: y / (1.0 + y**p) ** (1.0 / p))


def scalar_root(g, theta, delta, z_map, lo=-3.0, hi=3.0, steps=220):
    f = lambda lam: inner(theta, z_map(g + lam * theta, delta))
    assert f(lo) < 0.0 < f(hi), (f(lo), f(hi))
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def monotone_root(fun, target, hi=100.0):
    lo = 0.0
    for _ in range(240):
        mid = (lo + hi) / 2.0
        if fun(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def check_close(name, got, want, tol):
    err = abs(got - want)
    if err > tol:
        raise AssertionError(f"{name}: got {got}, want {want}, error {err}")
    print(f"PASS  {name:<56} error={err:.3e}")


def check_centers_and_tails():
    # Strict corank-two line: Huber selects K_F, pseudo-Huber selects K_R.
    astar = np.diag([1.0, 1.0, 0.0, 0.0])
    theta = np.diag([3 / 8, 3 / 8, 1.0, 0.5])
    g = astar - theta
    sig = np.array([1.0, 0.5])
    a = 0.75
    eta_f = monotone_root(
        lambda e: sum(s * min(e * s, 1.0) for s in sig), a
    )
    eta_r = monotone_root(
        lambda e: sum(e * s * s / math.sqrt(1.0 + e * e * s * s) for s in sig),
        a,
    )
    phi_f = np.diag([1.0, 1.0, -eta_f, -0.5 * eta_f])
    phi_r = np.diag(
        [
            1.0,
            1.0,
            -eta_r / math.sqrt(1.0 + eta_r * eta_r),
            -0.5 * eta_r / math.sqrt(1.0 + 0.25 * eta_r * eta_r),
        ]
    )
    delta = 1e-5
    lph = scalar_root(g, theta, delta, z_ph)
    lhu = scalar_root(g, theta, delta, z_huber)
    check_close("pseudo-Huber strict slope", (lph - 1.0) / delta, -eta_r, 2e-8)
    check_close("Huber strict slope", (lhu - 1.0) / delta, -eta_f, 2e-8)
    check_close(
        "pseudo-Huber center K_R", np.linalg.norm(z_ph(g + lph * theta, delta) - phi_r), 0.0, 2e-9
    )
    check_close(
        "Huber center K_F", np.linalg.norm(z_huber(g + lhu * theta, delta) - phi_f), 0.0, 2e-9
    )
    assert np.linalg.norm(phi_f - phi_r) > 0.04

    # Boundary line from the paper.
    theta2 = np.array([[1.0, 0.0], [0.0, 0.0]])
    g2 = np.array([[0.0, 1.0], [1.0, 1.0]])
    beta = kappa = 0.5

    delta = 1e-9
    lph = scalar_root(g2, theta2, delta, z_ph)
    cph = (2.0 * beta * kappa) ** (-1.0 / 3.0)
    check_close("pseudo-Huber p=2 tail constant", (1.0 - lph) / delta ** (2 / 3), cph, 2e-6)

    # A non-pseudo-Huber profile with p=3 and gamma=1/3.
    p = 3.0
    gamma = 1.0 / p
    lp3 = scalar_root(g2, theta2, delta, lambda x, d: z_power_tail(x, d, p))
    cp3 = (gamma / (kappa * beta ** (p - 1.0))) ** (1.0 / (p + 1.0))
    check_close("general p=3 tail constant", (1.0 - lp3) / delta ** (p / (p + 1.0)), cp3, 1e-6)

    # Finite saturation: the constant is L/beta, not universally 1/beta.
    for threshold in (1.0, 0.5):
        lhu = scalar_root(
            g2, theta2, delta, lambda x, d, ell=threshold: z_huber(x, d, ell)
        )
        check_close(
            f"finite-saturation constant L/beta (L={threshold:g})",
            (1.0 - lhu) / delta,
            threshold / beta,
            3e-6,
        )

    # Huber may have a whole multiplier interval although its certificate is unique.
    g3 = 2.0 * np.eye(2)
    theta3 = np.diag([1.0, -1.0])
    for lam in (-1.0, 0.0, 1.0):
        h = inner(theta3, z_huber(g3 + lam * theta3, 0.1))
        check_close(f"Huber multiplier interval at lambda={lam:+.0f}", h, 0.0, 1e-14)


def mp_diag(values):
    out = mp.matrix(len(values))
    for i, value in enumerate(values):
        out[i, i] = value
    return out


def z_ph_mp(x, delta):
    gram = x.T * x + delta * delta * mp.eye(x.cols)
    eigvals, q = mp.eigsy(gram)
    invroot = q * mp_diag([1 / mp.sqrt(eigvals[i]) for i in range(len(eigvals))]) * q.T
    return x * invroot


def inner_mp(x, y):
    return mp.fsum(x[i, j] * y[i, j] for i in range(x.rows) for j in range(x.cols))


def root_mp(g, theta, delta, lo=mp.mpf("0.5"), hi=mp.mpf("1.5"), steps=420):
    def f(lam):
        return inner_mp(theta, z_ph_mp(g + lam * theta, delta))

    assert f(lo) < 0 < f(hi)
    for _ in range(steps):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def frobenius_mp(x):
    return mp.sqrt(mp.fsum(x[i, j] ** 2 for i in range(x.rows) for j in range(x.cols)))


def check_quintic_branch():
    mp.mp.dps = 100
    astar = mp_diag([3, 2, 1, 0])
    theta = mp.matrix(
        [
            [mp.mpf(27) / 10, 0, 0, mp.mpf(3) / 10],
            [0, -mp.mpf(16) / 5, 0, mp.mpf(1) / 10],
            [0, 0, mp.mpf(1) / 2, mp.mpf(1) / 4],
            [mp.mpf(1) / 5, -mp.mpf(3) / 20, mp.mpf(1) / 10, 1],
        ]
    )
    g = astar - theta
    a = theta[0, 0] + theta[1, 1] + theta[2, 2]
    chi_star = theta[0, 0] / 9 + theta[1, 1] / 4 + theta[2, 2]
    chi_2 = theta[0, 0] / 81 + theta[1, 1] / 16 + theta[2, 2]
    invariant_tol = mp.mpf("1e-95")
    assert abs(a) < invariant_tol
    assert abs(chi_star) < invariant_tol
    assert abs(chi_2 - mp.mpf(1) / 3) < invariant_tol
    expected = {
        "1e-2": mp.mpf("-0.124820736116491"),
        "1e-3": mp.mpf("-0.124983349780391"),
        "1e-4": mp.mpf("-0.124998347562115"),
        "1e-5": mp.mpf("-0.124999834881873"),
    }
    last_phi = None
    last_delta = None
    ratios = []
    for text_delta, target in expected.items():
        delta = mp.mpf(text_delta)
        lam = root_mp(g, theta, delta)
        ratio = (lam - 1) / delta**5
        ratios.append(ratio)
        err = abs(ratio - target)
        if err > mp.mpf("5e-14"):
            raise AssertionError(f"quintic coefficient {text_delta}: {ratio} vs {target}")
        print(f"PASS  quintic coefficient delta={text_delta:<5}              error={mp.nstr(err, 3)}")
        if text_delta == "1e-4":
            last_phi = z_ph_mp(g + lam * theta, delta)
            last_delta = delta

    limit_errors = [abs(ratio + mp.mpf(1) / 8) for ratio in ratios]
    if not all(limit_errors[i + 1] < limit_errors[i] for i in range(len(limit_errors) - 1)):
        raise AssertionError(f"quintic errors are not decreasing: {limit_errors}")
    if limit_errors[-1] > mp.mpf("2e-7"):
        raise AssertionError(f"quintic limit -1/8 not resolved: {ratios[-1]}")
    print(f"PASS  quintic ratios converge to -1/8                       error={mp.nstr(limit_errors[-1], 4)}")

    phi_star = mp_diag([1, 1, 1, 0])
    jstar = mp_diag([mp.mpf(1) / 9, mp.mpf(1) / 4, 1, 0])
    observed = (last_phi - phi_star + last_delta**2 * jstar / 2) / last_delta**4
    predicted = mp_diag([mp.mpf(1) / 216, mp.mpf(3) / 128, mp.mpf(3) / 8, -mp.mpf(1) / 8])
    err = frobenius_mp(observed - predicted)
    if err > mp.mpf("1e-5"):
        raise AssertionError(f"quartic direction coefficient: error {err}")
    print(f"PASS  quartic direction matrix                              error={mp.nstr(err, 4)}")
    offdiag = mp.sqrt(
        mp.fsum(
            observed[i, j] ** 2
            for i in range(observed.rows)
            for j in range(observed.cols)
            if i != j
        )
    )
    if offdiag > mp.mpf("4.5e-6"):
        raise AssertionError(f"quartic off-diagonal residual: {offdiag}")
    print(f"PASS  quartic off-diagonal residual                         norm={mp.nstr(offdiag, 4)}")


if __name__ == "__main__":
    check_centers_and_tails()
    check_quintic_branch()
    print("ALL SMOOTHING-FAMILY ASSERTIONS PASSED")
