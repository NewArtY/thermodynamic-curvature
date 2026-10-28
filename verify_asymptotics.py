# -*- coding: utf-8 -*-
r"""
verify_asymptotics.py -- exact asymptotic series of the closed forms (D = 3)
============================================================================

    c1(zeta) = Xi(zeta) K_2(2 zeta) / (2 zeta^2 K_2(zeta)^2)            (dilute BG curvature)
    Y(zeta)  = (1/4) [2 c2 + c4/c2 - c3^2/c2^2]                         (O(kappa^2) Kaniadakis)

NR (zeta -> inf): Hankel series K_nu(x) ~ sqrt(pi/2x) e^{-x} sum_k a_k(nu) x^{-k}.
UR (zeta -> 0) : convergent series of K_n with logarithms.
Each series is checked against mpmath at a test point.  Output data/asymptotics.json.
"""
import json
import os

import mpmath as mp
import sympy as sp

import thermogeom as tg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "asymptotics.json")
NTERMS = 9

eps = sp.symbols("epsilon", positive=True)      # 1/zeta
x = sp.symbols("x", positive=True)              # zeta (UR)
L = sp.symbols("L")                              # ln(zeta/2)


def hankel_S(nu, xinv, n=NTERMS):
    """sum_k a_k(nu) x^{-k}."""
    s = 0
    for k in range(n):
        num = 1
        for j in range(1, k + 1):
            num *= (4 * nu ** 2 - (2 * j - 1) ** 2)
        s += sp.Rational(num, sp.factorial(k) * 8 ** k) * xinv ** k
    return s


def cumulant_exprs(zeta, R):
    """c1..c4 as expressions in (zeta, R) using thermogeom's sympy generator (D=3)."""
    _, polys = tg._cumulant_polys(3)
    zs, Rs = sp.symbols("zeta R")
    return [p.subs({zs: zeta, Rs: R}) for p in polys[1:]]


def nr_series():
    z = 1 / eps
    R1 = sp.series(hankel_S(1, eps) / hankel_S(2, eps), eps, 0, NTERMS).removeO()
    R2 = sp.series(hankel_S(1, eps / 2) / hankel_S(2, eps / 2), eps, 0, NTERMS).removeO()
    c1a, c2a, c3a, c4a = [sp.series(sp.expand(c), eps, 0, 5).removeO() for c in cumulant_exprs(z, R1)]
    c1b, c2b, _, _ = [sp.series(sp.expand(c), eps, 0, 5).removeO() for c in cumulant_exprs(z / 2 * 2 * 2 / 2 * 2 / 2, R2)] \
        if False else [sp.series(sp.expand(c), eps, 0, 5).removeO() for c in cumulant_exprs(2 * z, R2)]
    d = c1a - c1b / 2
    Xi = sp.series(c2b / (4 * c2a) + d ** 2 / c2a + c3a * d / c2a ** 2 - 1, eps, 0, 4).removeO()
    pref = sp.series(hankel_S(2, eps / 2) / hankel_S(2, eps) ** 2, eps, 0, 4).removeO()
    c1ser = sp.expand(sp.series(Xi * pref, eps, 0, 4).removeO())          # times zeta^{-3/2}/(2 sqrt(pi))
    Y = sp.series((2 * c2a + c4a / c2a - c3a ** 2 / c2a ** 2) / 4, eps, 0, 4).removeO()
    return sp.expand(Xi), c1ser, sp.expand(Y)


def K_small(n, xx, Lx, order=8):
    """Series of K_n(xx) for integer n, with Lx = ln(xx/2) as an independent symbol."""
    s = sp.Rational(1, 2) * (xx / 2) ** (-n) * sum(sp.factorial(n - k - 1) / sp.factorial(k) * (-xx ** 2 / 4) ** k
                                                   for k in range(n))
    In = sum((xx / 2) ** (n + 2 * k) / (sp.factorial(k) * sp.factorial(n + k)) for k in range(order))
    s += (-1) ** (n + 1) * Lx * In
    psi = lambda m: -sp.EulerGamma + sp.harmonic(m - 1)      # psi(m)
    s += (-1) ** n * sp.Rational(1, 2) * (xx / 2) ** n * sum(
        (psi(k + 1) + psi(n + k + 1)) * (xx ** 2 / 4) ** k / (sp.factorial(k) * sp.factorial(n + k))
        for k in range(order))
    return sp.expand(s)


def ur_series(order=8):
    K1a, K2a = K_small(1, x, L), K_small(2, x, L)
    K1b, K2b = K_small(1, 2 * x, L + sp.log(2)), K_small(2, 2 * x, L + sp.log(2))
    Ra = sp.series(K1a / K2a, x, 0, order).removeO()
    Rb = sp.series(K1b / K2b, x, 0, order).removeO()
    c1a, c2a, c3a, c4a = [sp.expand(sp.series(sp.expand(c), x, 0, 6).removeO()) for c in cumulant_exprs(x, Ra)]
    c1b, c2b, _, _ = [sp.expand(sp.series(sp.expand(c), x, 0, 6).removeO()) for c in cumulant_exprs(2 * x, Rb)]
    d = c1a - c1b / 2
    Xi = sp.expand(sp.series(c2b / (4 * c2a) + d ** 2 / c2a + c3a * d / c2a ** 2 - 1, x, 0, 5).removeO())
    pref = sp.expand(sp.series(K2b / (2 * x ** 2 * K2a ** 2), x, 0, 5).removeO())
    c1ser = sp.expand(sp.series(Xi * pref, x, 0, 5).removeO())
    Y = sp.expand(sp.series((2 * c2a + c4a / c2a - c3a ** 2 / c2a ** 2) / 4, x, 0, 5).removeO())
    return Xi, c1ser, Y


def main():
    out = {}
    Xi_nr, c1_nr, Y_nr = nr_series()
    print("NR:  Xi(zeta)  =", Xi_nr)
    print("NR:  c1(zeta)  = 5/(16 sqrt(pi)) zeta^{-3/2} * [", sp.expand(c1_nr / sp.Rational(5, 8)), "]")
    print("NR:  Y(zeta)   =", Y_nr)
    out["NR"] = {"Xi": str(Xi_nr), "c1_bracket": str(sp.expand(c1_nr / sp.Rational(5, 8))), "Y": str(Y_nr)}
    # numerical check at zeta = 200
    z0 = 200
    with mp.workdps(40):
        c1n = tg.c1_closed(z0, dps=40)
        approx = sp.N(c1_nr.subs(eps, sp.Rational(1, z0)) / (2 * sp.sqrt(sp.pi)) * sp.Integer(z0) ** sp.Rational(-3, 2), 30)
        Yn = tg.Y_kappa(z0, dps=40)
        Ya = sp.N(Y_nr.subs(eps, sp.Rational(1, z0)), 30)
        print("    check zeta=200: c1 exact %s  series %s ;  Y exact %s series %s" %
              (mp.nstr(c1n, 15), sp.N(approx, 15), mp.nstr(Yn, 15), sp.N(Ya, 15)))
        out["NR_check_zeta200"] = [float(c1n), float(approx), float(Yn), float(Ya)]

    Xi_ur, c1_ur, Y_ur = ur_series()
    print("UR:  Xi(zeta) =", sp.collect(Xi_ur, x))
    print("UR:  c1(zeta) =", sp.collect(c1_ur, x))
    print("UR:  Y(zeta)  =", sp.collect(Y_ur, x))
    out["UR"] = {"Xi": str(Xi_ur), "c1": str(c1_ur), "Y": str(Y_ur)}
    for z0 in (sp.Rational(1, 100), sp.Rational(1, 20)):
        with mp.workdps(40):
            c1n = tg.c1_closed(float(z0), dps=40)
            Yn = tg.Y_kappa(float(z0), dps=40)
        sub = {x: z0, L: sp.log(z0 / 2)}
        print("    check zeta=%s: c1 exact %s series %s ; Y exact %s series %s" %
              (z0, mp.nstr(c1n, 15), sp.N(c1_ur.subs(sub), 15), mp.nstr(Yn, 15), sp.N(Y_ur.subs(sub), 15)))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
