# -*- coding: utf-8 -*-
"""closed_forms_print.py -- print the explicit closed forms used in the article (D=3 cumulants; exact D=2 c1, Y)."""
import sympy as sp
import thermogeom as tg

_, c = tg._cumulant_polys(3)
z, R = sp.symbols("zeta R")
for k in range(1, 5):
    print("D=3  c_%d(zeta, R) =" % k, sp.collect(sp.expand(c[k]), [R, z]))

# exact D = 2: A(beta) = int p e^{-beta eps} dp = (1 + beta m) e^{-beta m}/beta^2
b, m = sp.symbols("beta m", positive=True)
A = (1 + b * m) * sp.exp(-b * m) / b ** 2
for a in (1,):
    B = -sp.Rational(a, 2) * A.subs(b, 2 * b)
    Ad = [sp.diff(A, b, k) for k in range(5)]
    Bd = [sp.diff(B, b, k) for k in range(3)]
    G = Ad[0] * Ad[2] - Ad[1] ** 2
    R0 = -(4 * (Ad[2] ** 2 - Ad[1] * Ad[3]) * Bd[0] + 2 * (Ad[0] * Ad[3] - Ad[1] * Ad[2]) * Bd[1] - G * Bd[2]) / (2 * G ** 2)
    c1_2D = sp.factor(sp.simplify(-R0.subs({b: 1, m: z}) / a))
    print("D=2  c1(zeta) =", c1_2D, "   c1(2) =", c1_2D.subs(z, 2), "  c1(1/2) =", c1_2D.subs(z, sp.Rational(1, 2)),
          "  c1(0) =", sp.limit(c1_2D, z, 0))
    M = [(-1) ** k * Ad[k] for k in range(5)]
    H2 = M[0] * M[2] - M[1] ** 2
    H3 = sp.Matrix([[M[0], M[1], M[2]], [M[1], M[2], M[3]], [M[2], M[3], M[4]]]).det()
    Y2 = sp.factor(sp.simplify((b ** 2 * A * H3 / (4 * H2 ** 2)).subs({b: 1, m: z})))
    print("D=2  Y(zeta)  =", Y2, "   Y(0) =", sp.limit(Y2, z, 0), "  Y(inf) =", sp.limit(Y2, z, sp.oo))
