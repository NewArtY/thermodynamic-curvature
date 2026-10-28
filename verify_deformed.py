# -*- coding: utf-8 -*-
r"""
verify_deformed.py -- Tsallis and Kaniadakis gases: claims C5-C9 and results N4, N5
===================================================================================

C5  existence: g_bb < inf  <=>  E/V < inf  <=>  P/T < inf  (same tail integral p^{D - P}),
    P = 1/kappa or 1/(q-1); bound P > D + 1 (relativistic), 2P > D + 2 (NR dispersion);
    escort Tsallis: q < 1 + 1/D.  Cut-off demonstration in D = 3.
C6  classical Tsallis flat: plain determinant at 50 digits on a grid (q, zeta, alpha, D).
C7  Tsallis quantum sign rule (dense grid, fast path + mp spot checks).
C8  classical Kaniadakis: R < 0 (dense grid).
C9  Kaniadakis Bose: sign changes, crossover ztilde*(kappa) at zeta = 0.1, 1, 10 (mp roots);
    Kaniadakis Fermi sign.
N4  small-kappa law  R_kappa^cl = -kappa^2 Y(zeta)/n + O(kappa^4) and ztilde* ~ kappa^2 Y/(c1 A).
N5  deep-dilute laws at fixed deformation (mp, alpha up to 1e4):
      Tsallis quantum:   R -> -a Cq(q) beta^3 gamma'^{-4},  gamma' = gamma + 1/(q-1),
                         Cq = (p-1)^2 (p-2) (p-3)^2 (p-4) / (4 (2p-1)(2p-3)),  p = 1/(q-1)
      Kaniadakis (any a): R -> -Kk(kappa) beta^3 gamma^{1/kappa - 6},
                         Kk = (2 kappa)^P (P-1)(P-2)(P-3)^2(P-4)/(P+1),  P = 1/kappa
    (reduced units, D = 3; massless constants, approached by the massive gas for gamma >> zeta).
Outputs data/deformed_results.json, data/deformed_curves.npz
"""
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import mpmath as mp
import numpy as np

import thermogeom as tg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = {}
NPROC = max(1, (os.cpu_count() or 2) - 2)


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100, flush=True)


# ------------------------------------------------------------------ workers (picklable)
def _w_curv(args):
    kind, a, par, zeta, alpha, dps, red = args
    with mp.workdps(dps + 20):
        R = tg.curvature_mp(zeta, mp.mpf(alpha), tg.Stat(kind, a, par), dps=dps, red=red)
        return mp.nstr(R, dps)


def _w_plain(args):
    kind, a, par, zeta, alpha, D = args
    det, big, _ = tg.plain_det_scale_mp(zeta, alpha, tg.Stat(kind, a, par), D=D, dps=50)
    return float(abs(det) / big)


def pmap(fn, items):
    with ProcessPoolExecutor(NPROC) as ex:
        return list(ex.map(fn, items, chunksize=1))


def Cq_tsallis(q):
    p = 1.0 / (q - 1.0)
    return (p - 1) ** 2 * (p - 2) * (p - 3) ** 2 * (p - 4) / (4 * (2 * p - 1) * (2 * p - 3))


def Kk_kaniadakis(kappa):
    P = mp.mpf(1) / mp.mpf(kappa)
    return mp.power(2 * mp.mpf(kappa), P) * (P - 1) * (P - 2) * (P - 3) ** 2 * (P - 4) / (P + 1)


# -----------------------------------------------------------------------------
def C5_existence():
    hdr("C5  existence: tail integrals with a momentum cut-off Lambda (D = 3, zeta = 1, alpha = 1)")
    out = []
    with mp.workdps(30):
        for kind, par in (("k", 0.2), ("k", 0.25), ("k", 0.3), ("q", 1.2), ("q", 1.25), ("q", 1.3)):
            st = tg.Stat(kind, 0, par)
            m, al = mp.mpf(1), mp.mpf(1)

            def integrands(p):
                e = p * p / (mp.sqrt(p * p + m * m) + m)
                y = e + al
                n, w, wp, K = tg.kernels_mp(y, st, "none")
                if kind == "q":
                    F = mp.power(1 + (par - 1) * y, (par - 2) / (par - 1)) / (2 - par)
                else:
                    u = mp.exp(-mp.asinh(par * y) / par)
                    F = u * (mp.sqrt(1 + (par * y) ** 2) + par * par * y) / (1 - par * par)
                return [p ** 2 * F, p ** 2 * (e + m) * n, p ** 2 * e * e * w]

            row = []
            for L in (1e2, 1e4, 1e6, 1e8):
                vals = [mp.quad(lambda s, i=i: integrands(mp.exp(s))[i] * mp.exp(s),
                                [mp.log(1e-8), 0, mp.log(10), mp.log(L)]) for i in range(3)]
                row.append([float(v) for v in vals])
            print("%-12s %s=%-5g  phi(L) E(L) g_bb(L) at L=1e2,1e4,1e6,1e8:" % (
                {"k": "Kaniadakis", "q": "Tsallis"}[kind], "kappa" if kind == "k" else "q", par))
            for L, r in zip((1e2, 1e4, 1e6, 1e8), row):
                print("      L=%.0e   phi=%.8e   E=%.8e   g_bb=%.8e" % (L, r[0], r[1], r[2]))
            out.append(dict(kind=kind, par=par, Lambda=[1e2, 1e4, 1e6, 1e8], phi_E_gbb=row))
    RES["C5"] = out


# -----------------------------------------------------------------------------
def C6_tsallis_flat():
    hdr("C6  classical Tsallis: plain determinant / largest term (50 digits)")
    items = []
    for D in (2, 3):
        qmax = 1 + 1 / (D + 1)
        for q in (1.01, 1.1, 1.2, min(1.24, qmax - 0.01)):
            for zeta in (0.0, 0.5, 5.0):
                for al in (0.01, 1.0, 10.0, 100.0):
                    items.append(("q", 0, q, zeta, al, D))
    t0 = time.time()
    ratios = pmap(_w_plain, items)
    print("points: %d ; max |det|/max|term| = %.2e  (%.0fs)" % (len(items), max(ratios), time.time() - t0))
    RES["C6"] = dict(points=len(items), max_det_over_term=max(ratios))


# -----------------------------------------------------------------------------
def dense_alpha():
    return np.concatenate([np.logspace(-10, -1, 120), np.linspace(0.1, 27.7, 140)[1:],
                           np.logspace(math.log10(28), 3, 60)])


def C7_C8_signs():
    hdr("C7/C8  sign checks on dense grids (fast path) with mp spot checks")
    al = dense_alpha()
    curves = {"alpha": al}
    summ = []
    specs = []
    for q in (1.01, 1.05, 1.1, 1.15, 1.2, 1.24):
        for a in (1, -1):
            specs.append(("q", a, q))
    for k in (0.01, 0.05, 0.1, 0.15, 0.2, 0.24):
        specs.append(("k", 0, k))
        specs.append(("k", 1, k))
    for kind, a, par in specs:
        st = tg.Stat(kind, a, par)
        for zeta in (0.1, 1.0, 10.0):
            R = np.array([tg.curvature_np(zeta, x, st) for x in al])
            curves["%s_%d_%g_%g" % (kind, a, par, zeta)] = R
            sgn_expect = -1 if (a == 1 or kind == "k") else +1
            ok = bool(np.all(np.sign(R) == sgn_expect))
            idx = list(range(0, len(al), 16))
            spots = pmap(_w_curv, [(kind, a, par, zeta, float(al[i]), 25, None) for i in idx])
            spot_ok = all((float(s) > 0) == (sgn_expect > 0) for s in spots)
            rel = max(abs(R[i] / float(s) - 1) for i, s in zip(idx, spots))
            summ.append(dict(stat=st.label(), zeta=zeta, expected_sign=sgn_expect, all_fast_ok=ok,
                             all_spots_ok=spot_ok, min=float(R.min()), max=float(R.max()),
                             max_spot_relerr=float(rel)))
            print("%-28s zeta=%5g expected sign %+d: fast-grid ok=%s  mp-spots ok=%s  range [%.3e, %.3e]  "
                  "fast/mp max rel %.1e" % (st.label(), zeta, sgn_expect, ok, spot_ok, R.min(), R.max(), rel),
                  flush=True)
    RES["C7_C8"] = summ
    np.savez(os.path.join(DATA, "deformed_curves.npz"), **curves)


# -----------------------------------------------------------------------------
def C9_crossover():
    hdr("C9  Kaniadakis Bose: sign changes and crossover ztilde*(kappa)")
    kappas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.24]
    al = np.concatenate([np.logspace(-8, -1, 60), np.linspace(0.1, 40, 400)[1:], np.logspace(math.log10(41), 3, 80)])
    table = []
    for zeta in (0.1, 1.0, 10.0):
        for k in kappas:
            st = tg.Stat("k", -1, k)
            R = np.array([tg.curvature_np(zeta, x, st) for x in al])
            sc = tg.find_sign_changes(al, R)
            roots = []
            for i in sc:
                lo, hi = al[i], al[i + 1]
                # mp refinement (bisection + secant) at 30 digits
                with mp.workdps(40):
                    f = lambda x: tg.curvature_mp(zeta, x, st, dps=30)
                    r = mp.findroot(f, (mp.mpf(lo), mp.mpf(hi)), solver="illinois", tol=mp.mpf(10) ** -24)
                roots.append(float(r))
            zt = [math.exp(-r) for r in roots]
            with mp.workdps(30):
                lead = float(tg.zstar_small_kappa(k, zeta))
            zz = [z_ * math.exp(zeta) for z_ in zt]
            print("zeta=%5g kappa=%.2f: #sign changes=%d  alpha*=%s  ztilde*=%s  z*=%s  ztilde*/kappa^2=%s  "
                  "leading-order ztilde*=%.5f" % (zeta, k, len(sc), ["%.8f" % r for r in roots],
                                                  ["%.6f" % v for v in zt], ["%.5f" % v for v in zz],
                                                  ["%.4f" % (v / k ** 2) for v in zt], lead), flush=True)
            table.append(dict(zeta=zeta, kappa=k, n_sign_changes=len(sc), alpha_star=roots, ztilde_star=zt,
                              z_star=zz, ztilde_star_over_kappa2=[v / k ** 2 for v in zt],
                              leading_order_ztilde_star=lead, R_dilute_end=float(R[-1]), R_dense_end=float(R[0])))
    RES["C9"] = table
    # Kaniadakis Fermi sign (dense grid also includes deep values up to alpha=1e3)
    hdr("C9b  Kaniadakis Fermi sign")
    ok_all = True
    for zeta in (0.1, 1.0, 10.0):
        for k in kappas:
            R = np.array([tg.curvature_np(zeta, x, tg.Stat("k", 1, k)) for x in al])
            ok = bool(np.all(R < 0))
            ok_all &= ok
            if not ok:
                print("   FD sign violation zeta=%g kappa=%g" % (zeta, k))
    print("Kaniadakis Fermi negative on all grids:", ok_all)
    RES["C9_fermi_negative"] = ok_all


# -----------------------------------------------------------------------------
def N4_small_kappa():
    hdr("N4  O(kappa^2): R_kappa^cl * n / kappa^2  ->  -Y(zeta)   (n = ztilde A_kin)")
    out = []
    for zeta in (0.1, 1.0, 10.0):
        with mp.workdps(40):
            Y = tg.Y_kappa(zeta, dps=40)
            Akin = tg.A_reduced(zeta, kinetic=True, dps=40)
        items = [("k", 0, k, zeta, al, 40, None) for k in (1e-4, 1e-3, 1e-2, 3e-2) for al in (0.5, 2.0, 5.0)]
        Rs = pmap(_w_curv, items)
        print("zeta=%g: Y(zeta)=%s" % (zeta, mp.nstr(Y, 15)))
        for (kind, a, k, z, al, dps, red), Rstr in zip(items, Rs):
            with mp.workdps(40):
                n = mp.exp(-al) * Akin
                ratio = mp.mpf(Rstr) * n / mp.mpf(k) ** 2
            print("    kappa=%-7g alpha=%-4g  R n/kappa^2 = %s   (rel. dev. from -Y: %.2e)" %
                  (k, al, mp.nstr(ratio, 12), float(ratio / (-Y) - 1)))
            out.append(dict(zeta=zeta, kappa=k, alpha=al, R_n_over_kappa2=float(ratio), minus_Y=float(-Y)))
    RES["N4"] = out


# -----------------------------------------------------------------------------
def N5_deep_dilute():
    hdr("N5a Tsallis quantum at fixed q:  R gamma'^4 / beta^3 -> -a Cq(q)")
    out = {"tsallis": [], "kaniadakis": []}
    alphas = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    for q in (1.1, 1.2):
        Cq = Cq_tsallis(q)
        for zeta in (0.0, 1.0):
            for a in (1, -1):
                items = [("q", a, q, zeta, al_, 30, None) for al_ in alphas]
                Rs = pmap(_w_curv, items)
                vals = []
                for al_, Rstr in zip(alphas, Rs):
                    gp = al_ - zeta + 1 / (q - 1)            # gamma' = gamma + 1/(q-1), gamma = alpha - zeta
                    vals.append(float(mp.mpf(Rstr)) * gp ** 4)
                print("q=%.2f zeta=%g a=%+d: R gamma'^4 at alpha=%s:\n      %s   -> -a Cq = %.8f" %
                      (q, zeta, a, alphas, " ".join("%.8f" % v for v in vals), -a * Cq), flush=True)
                out["tsallis"].append(dict(q=q, zeta=zeta, a=a, alphas=alphas, R_gp4=vals, minus_a_Cq=-a * Cq))
    hdr("N5b Kaniadakis at fixed kappa:  R gamma^{6-1/kappa} / beta^3 -> -Kk(kappa)")
    alphas = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    for k in (0.24, 0.2, 1.0 / 6.0, 0.1, 0.05):
        Kk = Kk_kaniadakis(k)
        for zeta in (0.0, 1.0):
            for a in (0, -1, 1):
                items = [("k", a, k, zeta, al_, 30, None) for al_ in alphas]
                Rs = pmap(_w_curv, items)
                vals, raw = [], []
                for al_, Rstr in zip(alphas, Rs):
                    g_ = mp.mpf(al_) - zeta
                    raw.append(Rstr)
                    vals.append(float(mp.mpf(Rstr) * mp.power(g_, 6 - 1 / mp.mpf(k)) / Kk))
                print("kappa=%.4f zeta=%g a=%+d: R/(-Kk gamma^{1/k-6}) at alpha=%s:\n      %s ;  R = %s" %
                      (k, zeta, a, alphas, " ".join("%.6f" % (-v) for v in vals),
                       " ".join(mp.nstr(mp.mpf(r_), 5) for r_ in raw)), flush=True)
                out["kaniadakis"].append(dict(kappa=k, zeta=zeta, a=a, alphas=alphas,
                                              ratio_to_asymptote=[-v for v in vals], R=[str(r_) for r_ in raw],
                                              Kk=float(Kk)))
    RES["N5"] = out


def main():
    t0 = time.time()
    C5_existence()
    C6_tsallis_flat()
    N4_small_kappa()
    C9_crossover()
    N5_deep_dilute()
    C7_C8_signs()
    with open(os.path.join(DATA, "deformed_results.json"), "w") as fh:
        json.dump(RES, fh, indent=1, default=str)
    print("\ntotal %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
