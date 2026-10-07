"""Theorem 1 and the paradox, checked symbolically and numerically.

These tests are the guard on the analytical core: if a refactor changes
``kappa_crit``, the frozen predictions of every past experiment silently become
wrong, so the formulas are re-derived from scratch here rather than restated.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from ecfi.linear import (
    ILLUSTRATIVE, LocalParams, A_matrix, exactness_holds, is_schur, jury,
    kappa_crit, kappa_delta, paradox_holds, spectral_radius, u_star,
)


def test_kappa_crit_matches_symbolic_derivation():
    a, b, c, d, e, f, k, z = sp.symbols("a b c d e f kappa z", real=True)
    A = sp.Matrix([[a - c * k, b], [e + f * k, d]])
    poly = sp.expand(z**2 - A.trace() * z + A.det())
    pm1 = sp.expand(poly.subs(z, -1))
    root = sp.solve(sp.Eq(pm1, 0), k)[0]
    closed_form = ((1 + a) * (1 + d) - b * e) / (c * (1 + d) + b * f)
    assert sp.simplify(root - closed_form) == 0


def test_jury_quantities_match_paper():
    a, b, c, d, e, f, k = sp.symbols("a b c d e f kappa", real=True)
    tau = a - c * k + d
    delta = (a - c * k) * d - b * (e + f * k)
    p1 = sp.expand(1 - tau + delta)
    pm1 = sp.expand(1 + tau + delta)
    assert sp.simplify(p1 - ((1 - a) * (1 - d) - b * e + k * (c * (1 - d) - b * f))) == 0
    assert sp.simplify(pm1 - ((1 + a) * (1 + d) - b * e - (c * (1 + d) + b * f) * k)) == 0
    assert sp.simplify(delta - (a * d - b * e - k * (c * d + b * f))) == 0


def test_illustrative_regime_numbers():
    p = ILLUSTRATIVE
    kc = kappa_crit(p)
    assert kc == pytest.approx(1.671233, abs=1e-5)
    assert spectral_radius(p, kc) == pytest.approx(1.0, abs=1e-9)
    assert spectral_radius(p, 1.7) == pytest.approx(1.022550, abs=1e-5)
    assert kappa_delta(p) == pytest.approx(1.878788, abs=1e-5)
    assert exactness_holds(p) and paradox_holds(p)


def test_critical_eigenvalue_is_minus_one():
    p = ILLUSTRATIVE
    ev = np.linalg.eigvals(A_matrix(p, kappa_crit(p)))
    assert np.min(np.real(ev)) == pytest.approx(-1.0, abs=1e-9)


def test_bound_is_exact_under_the_stated_condition():
    p = ILLUSTRATIVE
    kc = kappa_crit(p)
    assert exactness_holds(p)
    for k in np.linspace(0.0, kc - 1e-6, 60):
        assert is_schur(p, float(k))
    for k in np.linspace(kc + 1e-4, 1.5 * kc, 40):
        assert not is_schur(p, float(k))


def test_pm1_is_the_first_jury_condition_to_fail():
    p = ILLUSTRATIVE
    assert kappa_crit(p) < kappa_delta(p)
    p1, pm1, delta = jury(p, kappa_crit(p))
    assert pm1 == pytest.approx(0.0, abs=1e-9)
    assert p1 > 0 and abs(delta) < 1


def test_paradox_static_quality_improves_while_stability_is_lost():
    p = ILLUSTRATIVE
    kc = kappa_crit(p)
    ks = np.linspace(0.0, kc * 0.99, 50)
    us = np.array([u_star(p, float(k)) for k in ks])
    rho = np.array([spectral_radius(p, float(k)) for k in ks])
    assert np.all(np.diff(us) < 0)          # equilibrium uncertainty falls
    assert rho[-1] > rho[0]                 # yet the system approaches the boundary
    assert u_star(p, 0.0) / u_star(p, kc) > 5


def test_ustar_denominator_is_p1():
    p = ILLUSTRATIVE
    for k in (0.0, 0.5, 1.2):
        p1, _, _ = jury(p, k)
        assert u_star(p, k, h=1.0) == pytest.approx((1 - p.d) / p1, rel=1e-12)


def test_validate_rejects_degenerate_parameters():
    with pytest.raises(ValueError):
        LocalParams(a=0.5, b=-0.1, c=0.5, d=0.5, e=0.5, f=0.5).validate()
