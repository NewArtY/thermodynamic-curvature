# -*- coding: utf-8 -*-
r"""
verify_quadrature.py -- numerical reliability of thermogeom.py
================================================================

(A) fast float64 path vs high-precision mpmath path (30 and 45 digits), and the
    symmetry-reduced determinant vs the plain determinant, over a grid of
    statistics (BG, Tsallis, Kaniadakis; a = +1, 0, -1), coldness and activity.
(B) Fully independent curvature: metric g_ij(beta, alpha) obtained by
    finite differences of the POTENTIAL phi (closed-form F, so no use of the
    w, w' kernels), then the Gaussian curvature from the Brioschi formula with
    finite-difference derivatives of the metric (no determinant formula), R = 2K.
    For deformed quantum statistics (no closed-form F) the metric is taken from
    the w-quadrature and only the curvature step is independent.
Outputs data/quadrature_check.json.
"""
import json
import os
import time

import mpmath as mp

import thermogeom as tg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "quadrature_check.json")


# ----------------------------------------------------------------------------
# (B) independent Brioschi curvature
# ----------------------------------------------------------------------------
def phi_mp(beta, alpha, m, st, D=3, dps=50):
    """Reduced potential phihat(beta, alpha) by exp-sinh quadrature of the closed-form F."""
    with mp.workdps(dps + 10):
        beta, alpha, m = mp.mpf(beta), mp.mpf(alpha), mp.mpf(m)
        p0 = mp.mpf(tg._p_scale(float(beta), float(alpha), float(m), st))
        halfpi = mp.pi / 2

        def F(y):
            if st.kind == "bg":
                u = mp.exp(-y)
                return u if st.a == 0 else mp.log1p(st.a * u) / st.a
            if st.kind == "q":
                q = mp.mpf(st.par)
                return mp.power(1 + (q - 1) * y, (q - 2) / (q - 1)) / (2 - q)
            k = mp.mpf(st.par)
            u = mp.exp(-mp.asinh(k * y) / k)
            return u * (mp.sqrt(1 + (k * y) ** 2) + k * k * y) / (1 - k * k)

        def f(t):
            p = p0 * mp.exp(halfpi * mp.sinh(t))
            e = p * p / (mp.sqrt(p * p + m * m) + m)
            return halfpi * mp.cosh(t) * p ** D * F(beta * e + alpha)

        thi = 4.0 if st.kind == "bg" else 6.2
        return mp.quad(f, mp.linspace(-4.6, thi, 12))


def metric_from_phi(beta, alpha, m, st, D, h_b, h_a, dps=50):
    """g_ij by central finite differences of phi (O(h^2))."""
    with mp.workdps(dps + 10):
        P = {}
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                P[(i, j)] = phi_mp(beta + i * h_b, alpha + j * h_a, m, st, D, dps)
        gbb = (P[(1, 0)] - 2 * P[(0, 0)] + P[(-1, 0)]) / h_b ** 2
        gaa = (P[(0, 1)] - 2 * P[(0, 0)] + P[(0, -1)]) / h_a ** 2
        gba = (P[(1, 1)] - P[(1, -1)] - P[(-1, 1)] + P[(-1, -1)]) / (4 * h_b * h_a)
        return gbb, gba, gaa


def metric_from_w(beta, alpha, m, st, D, dps=40):
    mo = tg.moments_mp(beta, alpha, m, st, D=D, dps=dps, red="none")
    I2, I1, I0 = mo["row1"]
    return I2, I1, I0


def brioschi_R(metric, beta, alpha, h_b, h_a):
    """R = 2K from the Brioschi formula with finite-difference metric derivatives."""
    M = {}
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            M[(i, j)] = metric(beta + i * h_b, alpha + j * h_a)
    E = {k: v[0] for k, v in M.items()}
    F = {k: v[1] for k, v in M.items()}
    G = {k: v[2] for k, v in M.items()}
    du = lambda X: (X[(1, 0)] - X[(-1, 0)]) / (2 * h_b)
    dv = lambda X: (X[(0, 1)] - X[(0, -1)]) / (2 * h_a)
    duu = lambda X: (X[(1, 0)] - 2 * X[(0, 0)] + X[(-1, 0)]) / h_b ** 2
    dvv = lambda X: (X[(0, 1)] - 2 * X[(0, 0)] + X[(0, -1)]) / h_a ** 2
    duv = lambda X: (X[(1, 1)] - X[(1, -1)] - X[(-1, 1)] + X[(-1, -1)]) / (4 * h_b * h_a)
    E0, F0, G0 = E[(0, 0)], F[(0, 0)], G[(0, 0)]
    Eu, Ev, Fu, Fv, Gu, Gv = du(E), dv(E), du(F), dv(F), du(G), dv(G)
    A = mp.matrix([[-dvv(E) / 2 + duv(F) - duu(G) / 2, Eu / 2, Fu - Ev / 2],
                   [Fv - Gu / 2, E0, F0],
                   [Gv / 2, F0, G0]])
    B = mp.matrix([[0, Ev / 2, Gu / 2], [Ev / 2, E0, F0], [Gu / 2, F0, G0]])
    K = (mp.det(A) - mp.det(B)) / (E0 * G0 - F0 ** 2) ** 2
    return 2 * K


def main():
    res = {"A_fast_vs_mp": [], "B_brioschi": []}
    cases = []
    for a in (1, -1):
        for zeta in (0.01, 1.0, 30.0):
            for al in (1e-10, 1e-3, 0.1, 1.0, 10.0, 30.0):
                cases.append((tg.Stat("bg", a), zeta, al))
    for q in (1.05, 1.15, 1.24):
        for a in (1, 0, -1):
            for al in (1e-6, 0.1, 1.0, 5.0, 20.0):
                cases.append((tg.Stat("q", a, q), 1.0, al))
    for k in (0.05, 0.15, 0.24):
        for a in (1, 0, -1):
            for al in (1e-6, 0.1, 1.0, 5.0, 20.0):
                cases.append((tg.Stat("k", a, k), 1.0, al))
    for k in (0.1, 0.2):
        for a in (0, -1):
            for zeta in (0.1, 10.0):
                cases.append((tg.Stat("k", a, k), zeta, 3.0))

    print("(A) fast vs mp  [relative differences]")
    print("%-26s %6s %8s %14s %10s %10s %10s %10s" % ("stat", "zeta", "alpha", "R_mp45", "fast", "mp30",
                                                       "plain30", "lev/red"))
    worst = 0.0
    for st, zeta, al in cases:
        t0 = time.time()
        Rf = tg.curvature_np(zeta, al, st)
        with mp.workdps(70):       # keep all digits of the mp results (no rounding on return)
            R30 = tg.curvature_mp(zeta, al, st, dps=30)
            R45 = tg.curvature_mp(zeta, al, st, dps=45)
            Rpl = tg.curvature_mp(zeta, al, st, dps=30, plain=True)
            scale = abs(R45) if R45 != 0 else mp.mpf(1)
            if st.a == 0 and st.kind in ("bg", "q"):
                scale = mp.mpf(1)        # exactly flat: report absolute values
            d_f = float(abs(mp.mpf(Rf) - R45) / scale)
            d_30 = float(abs(R30 - R45) / scale)
            d_pl = float(abs(Rpl - R45) / scale)
        red = tg.default_reduction(st, al)
        worst = max(worst, d_f)
        print("%-26s %6g %8.1e %14.6e %10.1e %10.1e %10.1e %10s  (%.1fs)" %
              (st.label(), zeta, al, float(R45), d_f, d_30, d_pl, red, time.time() - t0))
        res["A_fast_vs_mp"].append(dict(stat=st.label(), zeta=zeta, alpha=al, R_mp45=float(R45),
                                        relerr_fast=d_f, relerr_mp30=d_30, relerr_plain30=d_pl,
                                        reduction=red))
    print("worst fast-path relative error: %.2e" % worst)
    res["worst_fast_relerr"] = worst

    print("\n(B) independent Brioschi curvature from finite differences")
    bcases = [(tg.Stat("bg", 1), 1.0, 0.5, True), (tg.Stat("bg", -1), 1.0, 0.05, True),
              (tg.Stat("bg", -1), 0.0, 0.3, True), (tg.Stat("bg", 1), 5.0, 2.0, True),
              (tg.Stat("q", 0, 1.15), 1.0, 1.0, True), (tg.Stat("k", 0, 0.15), 1.0, 1.0, True),
              (tg.Stat("k", 0, 0.1), 2.0, 4.0, True),
              (tg.Stat("q", -1, 1.15), 1.0, 0.5, False), (tg.Stat("q", 1, 1.1), 1.0, 0.5, False),
              (tg.Stat("k", -1, 0.15), 1.0, 0.5, False), (tg.Stat("k", -1, 0.1), 1.0, 4.0, False)]
    for st, zeta, al, from_phi in bcases:
        t0 = time.time()
        beta, m = mp.mpf(1), mp.mpf(zeta)
        h_b = mp.mpf("1e-5")
        h_a = mp.mpf("1e-5") * max(1, al)
        if from_phi:
            hb2, ha2 = mp.mpf("1e-8"), mp.mpf("1e-8") * max(1, al)
            metric = lambda b, a_: metric_from_phi(b, a_, m, st, 3, hb2, ha2, dps=60)
        else:
            metric = lambda b, a_: metric_from_w(b, a_, m, st, 3, dps=40)
        with mp.workdps(60):
            Rb = brioschi_R(metric, beta, mp.mpf(al), h_b, h_a)
        Rd = tg.curvature_mp(zeta, al, st, dps=40)
        rel = float(abs(Rb - Rd) / max(abs(Rd), mp.mpf("1e-30")))
        print("%-26s zeta=%5g alpha=%6g  R_Brioschi=%.12e  R_det=%.12e  rel.diff=%.1e  (%s, %.0fs)" %
              (st.label(), zeta, al, float(Rb), float(Rd), rel,
               "metric from phi" if from_phi else "metric from w", time.time() - t0))
        res["B_brioschi"].append(dict(stat=st.label(), zeta=zeta, alpha=al, R_brioschi=float(Rb),
                                      R_det=float(Rd), reldiff=rel, metric_from_phi=from_phi))
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
