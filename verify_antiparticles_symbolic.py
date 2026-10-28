# -*- coding: utf-8 -*-
r"""
verify_antiparticles_symbolic.py -- sympy derivations for the particle+antiparticle gas
=======================================================================================

phi_tot(beta, gamma) = phi(beta, gamma) + phi(beta, -gamma)       (gamma = -mu/T)

A1  MJ with antiparticles, phi = 2 cosh(gamma) A(beta), GENERIC A:
        det = 8 cosh(gamma) A'(A A''' - A' A''),   detg = 4 [cosh^2 A A'' - sinh^2 A'^2]
        R   = -cosh(gamma) A'(A A''' - A' A'') / (4 [cosh^2 A A'' - sinh^2 A'^2]^2)
    -> NOT flat (R = 0 only if A' = 0 or A A''' = A' A'').
A2  Power law phi = c beta^{-s} h(gamma) with h EVEN (mu = 0):  R = -beta^s / (c (s+1) h(0)) = -1/((s+1) phi).
    General gamma: explicit formula in h, h', h'', h'''.
A3  Massless FD +- : h = 2[7 pi^4/360 + pi^2 x^2/12 + x^4/24], x = mu/T  -> closed-form R(x).
A4  Dilute (large zeta) expansion at mu = 0:
        phi = 2 cosh(g) A(b) - lam a cosh(2g) A(2b) + O(lam^2)
    -> R = R_MJ(+-) + lam a Delta + O(lam^2); Delta in terms of A^(k)(1), A^(k)(2).
    NR leading behaviour of R_MJ and Delta for A = b^{-3/2} e^{-b m}(1 + ...).
Writes data/antiparticles_symbolic.json
"""
import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "antiparticles_symbolic.json")

b, g = sp.symbols("beta gamma", real=True)


def Rdet(phi, x, y):
    d = lambda *a: sp.diff(phi, *a)
    M = sp.Matrix([[d(x, x), d(x, y), d(y, y)],
                   [d(x, x, x), d(x, x, y), d(x, y, y)],
                   [d(x, x, y), d(x, y, y), d(y, y, y)]])
    G = d(x, x) * d(y, y) - d(x, y) ** 2
    return M.det(), G


def main():
    out = {}
    # ---------------- A1
    A = sp.Function("A")(b)
    phi = 2 * sp.cosh(g) * A
    det, G = Rdet(phi, b, g)
    A0, A1, A2, A3 = A, A.diff(b), A.diff(b, 2), A.diff(b, 3)
    det_claim = 8 * sp.cosh(g) * A1 * (A0 * A3 - A1 * A2)
    G_claim = 4 * (sp.cosh(g) ** 2 * A0 * A2 - sp.sinh(g) ** 2 * A1 ** 2)
    ok1 = sp.simplify((det - det_claim).rewrite(sp.exp)) == 0
    ok2 = sp.simplify((G - G_claim).rewrite(sp.exp)) == 0
    print("A1 MJ+- det formula:", ok1, " detg formula:", ok2)
    out["A1_det"] = bool(ok1)
    out["A1_detg"] = bool(ok2)
    # massless MJ: A = 2 beta^-3 -> R(beta=1) = -cosh/(cosh^2+3)^2
    Am = 2 / b ** 3
    det_m, G_m = Rdet(2 * sp.cosh(g) * Am, b, g)
    Rm = sp.simplify((-det_m / (2 * G_m ** 2)).subs(b, 1).rewrite(sp.exp))
    C = sp.symbols("C", positive=True)
    ok3 = sp.simplify(Rm - (-sp.cosh(g) / (sp.cosh(g) ** 2 + 3) ** 2).rewrite(sp.exp)) == 0
    print("A1 massless MJ+-: R = -cosh g/(cosh^2 g + 3)^2 :", ok3)
    out["A1_massless"] = bool(ok3)

    # ---------------- A2
    s, c = sp.symbols("s c", positive=True)
    h = sp.Function("h")(g)
    phi = c * b ** (-s) * h
    det, G = Rdet(phi, b, g)
    Rgen = sp.simplify((-det / (2 * G ** 2)).subs(b, 1))
    H0, H1, H2, H3 = sp.symbols("h0 h1 h2 h3")
    Rgen_h = sp.simplify(Rgen.subs({h.diff(g, 3): H3}).subs({h.diff(g, 2): H2})
                         .subs({h.diff(g): H1}).subs({h: H0}))
    print("A2 power law, general gamma, beta=1:  R =", sp.factor(Rgen_h))
    out["A2_general"] = str(sp.factor(Rgen_h))
    R0 = sp.simplify(Rgen_h.subs({H1: 0, H3: 0}))
    print("A2 even h at gamma=0: R =", R0)
    out["A2_even"] = str(R0)

    # ---------------- A3
    x = sp.symbols("x", real=True)          # x = mu/T = -gamma
    hF = 7 * sp.pi ** 4 / 360 + sp.pi ** 2 * g ** 2 / 12 + g ** 4 / 24   # = f4(e^-g) + f4(e^g), FD
    RF = sp.factor(sp.simplify(Rgen_h.subs({H0: hF, H1: hF.diff(g), H2: hF.diff(g, 2),
                                             H3: hF.diff(g, 3), s: 3, c: 2})))
    RFx = sp.factor(sp.simplify(RF.subs(g, -x)))
    print("A3 massless FD+-: R(x=mu/T) =", RFx)
    print("    at x=0:", sp.nsimplify(RFx.subs(x, 0)), "=", sp.N(RFx.subs(x, 0), 15))
    print("    large x:", sp.limit(RFx * x ** 4, x, sp.oo))
    out["A3_FD_massless"] = str(RFx)
    out["A3_FD_massless_mu0"] = str(sp.simplify(RFx.subs(x, 0)))
    # cross-check against the polylog inversion: h = -Li4(-e^-g) - Li4(-e^g)
    import mpmath as mp
    for gv in (0.0, 0.7, -2.3):
        hv = -mp.polylog(4, -mp.e ** (-gv)) - mp.polylog(4, -mp.e ** gv)
        hc = float(hF.subs(g, gv))
        print("    inversion check g=%g: polylog %.15f  poly %.15f" % (gv, float(mp.re(hv)), hc))
    # single-species massless FD at mu = 0 (for comparison): h = f4(e^-g) one term
    # ---------------- A4
    lam, aa = sp.symbols("lambda a")
    Bf = sp.Function("B")(b)            # B(beta) = A(2 beta)
    phi = 2 * sp.cosh(g) * A - lam * aa * sp.cosh(2 * g) * Bf
    det, G = Rdet(phi, b, g)
    a_ = sp.symbols("a0:4")
    b_ = sp.symbols("b0:4")
    rep = {}
    for k in range(3, -1, -1):
        rep[A.diff(b, k) if k else A] = a_[k]
        rep[Bf.diff(b, k) if k else Bf] = b_[k]
    det0 = sp.expand(det.subs(g, 0).doit())
    G0 = sp.expand(G.subs(g, 0).doit())
    for k in range(3, -1, -1):
        det0 = det0.subs(A.diff(b, k) if k else A, a_[k]).subs(Bf.diff(b, k) if k else Bf, b_[k])
        G0 = G0.subs(A.diff(b, k) if k else A, a_[k]).subs(Bf.diff(b, k) if k else Bf, b_[k])
    R = -det0 / (2 * G0 ** 2)
    R0 = sp.simplify(R.subs(lam, 0))
    R1 = sp.simplify(sp.diff(R, lam).subs(lam, 0) / aa)
    print("A4 R_MJ(mu=0) =", sp.factor(R0))
    print("A4 Delta      =", sp.factor(R1))
    out["A4_R0"] = str(sp.factor(R0))
    out["A4_Delta"] = str(sp.factor(R1))
    # NR asymptotics: A(beta) = beta^{-3/2} e^{-beta m} (1 + 15/(8 beta m) + 105/(128 (beta m)^2))
    # (D=3: A = m^2 K_2(beta m)/beta ~ sqrt(pi/2) m^{3/2} beta^{-3/2} e^{-beta m}[1 + 15/(8 x) + 105/(128 x^2)])
    m = sp.symbols("m", positive=True)
    eps = sp.symbols("epsilon", positive=True)      # 1/m
    # Hankel series of K_2: sum_k prod_{j<=k}(16 - (2j-1)^2) / (k! 8^k x^k); NHANK terms are
    # needed because a0 a3 - a1 a2 cancels two leading orders in 1/m.
    NHANK = 8
    xx = b * m
    hank = sum(sp.prod([16 - (2 * j - 1) ** 2 for j in range(1, k + 1)]) / (sp.factorial(k) * 8 ** k)
               / xx ** k for k in range(NHANK))
    Aex = sp.sqrt(sp.pi / 2) * m ** sp.Rational(3, 2) * b ** sp.Rational(-3, 2) * sp.exp(-b * m) * hank
    vals = {}
    for k in range(4):
        vals[a_[k]] = sp.diff(Aex, b, k).subs(b, 1)
        vals[b_[k]] = sp.diff(Aex.subs(b, 2 * b), b, k).subs(b, 1)
    R0m = (R0.subs(vals) * sp.exp(-m) * m ** sp.Rational(7, 2)).subs(m, 1 / eps)
    ser0 = sp.series(sp.simplify(R0m), eps, 0, 4).removeO()
    print("A4 NR: R_MJ(mu=0) * m^{7/2} e^{-m} =", sp.expand(sp.simplify(ser0)))
    R1m = (R1.subs(vals) * m ** sp.Rational(3, 2)).subs(m, 1 / eps)
    ser1 = sp.series(sp.simplify(R1m), eps, 0, 4).removeO()
    print("A4 NR: Delta * m^{3/2} =", sp.simplify(ser1))
    out["A4_NR_RMJ_times_m72_em"] = str(sp.simplify(ser0))
    out["A4_NR_Delta_times_m32"] = str(sp.simplify(ser1))
    # ---------------- A5: mu = 0 (phi even in gamma): R = -(d_b ln chi)(d_b ln(g_bb/chi)) / (2 g_bb)
    Pf = sp.Function("P")(b)      # phi(beta, 0)
    Qf = sp.Function("Q")(b)      # phi_gg(beta, 0) = chi
    Uf = sp.Function("U")(b)      # phi_gggg(beta, 0)
    phi_even = Pf + Qf * g ** 2 / 2 + Uf * g ** 4 / 24
    det, G = Rdet(phi_even, b, g)
    R_det = sp.simplify((-det / (2 * G ** 2)).subs(g, 0))
    a_bb = Pf.diff(b, 2)
    R_chi = -(sp.diff(sp.log(Qf), b) * sp.diff(sp.log(a_bb / Qf), b)) / (2 * a_bb)
    ok5 = sp.simplify(R_det - R_chi) == 0
    print("A5 mu=0: R = -(d ln chi)(d ln(g_bb/chi))/(2 g_bb):", ok5)
    out["A5_mu0_susceptibility_form"] = bool(ok5)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.relpath(OUT, HERE).replace(os.sep, "/"))


if __name__ == "__main__":
    main()
