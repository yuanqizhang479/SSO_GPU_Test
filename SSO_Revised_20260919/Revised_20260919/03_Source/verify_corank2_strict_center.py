#!/usr/bin/env python3
"""Deterministic high-precision check of the corank-two strict instance.

Uses only Python's standard-library Decimal module.  Every reported claim is
guarded by an assertion; a failed check exits nonzero.
"""

# Scientific assertions must never disappear under python -O / PYTHONOPTIMIZE.
if not __debug__:
    raise RuntimeError("Verification requires assertions: rerun without -O / PYTHONOPTIMIZE.")

from decimal import Decimal, getcontext

getcontext().prec = 90
D = Decimal
ZERO = D(0)
ONE = D(1)
HALF = ONE / D(2)
THREE_EIGHTHS = D(3) / D(8)


def absd(x):
    return x.copy_abs()


def bisect_increasing(fun, lo, hi, tol=D("1e-75")):
    flo, fhi = fun(lo), fun(hi)
    assert flo < 0 < fhi
    while hi - lo > tol:
        mid = (lo + hi) / D(2)
        if fun(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / D(2)


def eta_equation(eta):
    # sigma(C)=(1,1/2), |a|=3/4.
    return (
        eta / (ONE + eta * eta).sqrt()
        + (eta / D(4)) / (ONE + eta * eta / D(4)).sqrt()
        - D(3) / D(4)
    )


ETA_R = bisect_increasing(eta_equation, ZERO, D(10))
KR = (
    -ETA_R / (ONE + ETA_R * ETA_R).sqrt(),
    -(ETA_R / D(2)) / (ONE + ETA_R * ETA_R / D(4)).sqrt(),
)
KF = (-D(3) / D(5), -D(3) / D(10))


def frobenius(values):
    return sum(x * x for x in values).sqrt()


def smooth_root(delta):
    # s=lambda-lambda*.  The four diagonal entries of A(lambda) are
    # 1+3s/8, 1+3s/8, s, and s/2.
    def h(s):
        big = ONE + THREE_EIGHTHS * s
        return (
            D(2) * THREE_EIGHTHS * big / (big * big + delta * delta).sqrt()
            + s / (s * s + delta * delta).sqrt()
            + (s / D(4)) / (s * s / D(4) + delta * delta).sqrt()
        )

    return bisect_increasing(h, -D("0.2"), ZERO)


def direction(s, delta):
    big = ONE + THREE_EIGHTHS * s
    return (
        big / (big * big + delta * delta).sqrt(),
        big / (big * big + delta * delta).sqrt(),
        s / (s * s + delta * delta).sqrt(),
        (s / D(2)) / (s * s / D(4) + delta * delta).sqrt(),
    )


PHI_R = (ONE, ONE, KR[0], KR[1])
PHI_F = (ONE, ONE, KF[0], KF[1])
CENTER_DISTANCE = frobenius((KR[0] - KF[0], KR[1] - KF[1]))

# Exact feasibility/KKT checks for both centers.
assert absd(D(3) / D(4) + KR[0] + KR[1] / D(2)) < D("1e-70")
assert absd(D(3) / D(4) + KF[0] + KF[1] / D(2)) < D("1e-70")
assert max(absd(x) for x in KR + KF) <= ONE
assert D("0.04099") < CENTER_DISTANCE < D("0.04101")
assert absd(ETA_R - D("0.7150800689800525")) < D("1e-15")

rows = []
for delta in map(D, ("1e-2", "1e-3", "1e-4", "1e-5")):
    s = smooth_root(delta)
    phi = direction(s, delta)
    err_r = frobenius(tuple(phi[i] - PHI_R[i] for i in range(4)))
    err_f = frobenius(tuple(phi[i] - PHI_F[i] for i in range(4)))
    tangent = (
        THREE_EIGHTHS * phi[0]
        + THREE_EIGHTHS * phi[1]
        + phi[2]
        + HALF * phi[3]
    )
    assert absd(tangent) < D("1e-65")
    assert max(absd(x) for x in phi) < ONE
    rows.append((delta, s, err_r, err_f))

# Path multiplier and center-selection assertions.
assert absd(rows[-1][1] / rows[-1][0] + ETA_R) < D("1e-8")
assert rows[0][2] > rows[1][2] > rows[2][2] > rows[3][2]
assert rows[-1][2] < D("1e-9")
assert absd(rows[-1][3] - CENTER_DISTANCE) < D("1e-8")
assert rows[-1][3] > D("0.04")

print("corank-two strict center check: PASS")
print(f"eta_R = {ETA_R}")
print(f"||K_F-K_R||_F = {CENTER_DISTANCE}")
print("delta, lambda_delta-lambda*, ||Phi_delta-Phi_R||_F, ||Phi_delta-Phi_F||_F")
for row in rows:
    print("  " + ", ".join(str(x) for x in row))
