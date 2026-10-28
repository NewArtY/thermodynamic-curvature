# -*- coding: utf-8 -*-
r"""
verify_bg.py -- Boltzmann-Gibbs branch (sigma = 0): claims C1-C4, C10 and results N2, N3
========================================================================================

C1  MJ (a=0) flat for all (beta, gamma), any D: plain 3x3 determinant at 50 digits.
C2  dilute limit Rhat -> -a c1(zeta): closed form vs old determinant vs direct limit; table
    for zeta in {0, .01, .1, .5, 1, 2, 4, 8, 16, 32, 100}; positivity & monotonicity; D = 2, 4.
N2  R_0 = -2 Xi(zeta) B_2(T) (second virial coefficient), Xi table; direct check of the
    general formula R0_from_virial on the exact A(beta), B(beta).
N3  UR / NR asymptotics; non-relativistic polylog (Janyszek-Mrugala) curvature vs the
    relativistic curvature at large zeta, and its z -> 0 limit -a 5 lambda^3/(2^{9/2} g).
C3  sign rule on a dense grid ztilde in [1e-12, 1-1e-10], zeta in {0.01,...,30}.
C4  BE divergence at ztilde -> 1: R ~ C_BE(zeta) alpha^{-1/2}, analytic C_BE; massless log law.
C10 virial: P/(nT) = 1 + a z 2^{-(D+1)/2} K_nu(2 zeta)/K_nu(zeta) + O(z^2), nu=(D+1)/2.
Outputs: data/bg_results.json, data/bg_curves.npz
"""
import json
import math
import os
import time

import mpmath as mp
import numpy as np

import thermogeom as tg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = {}


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


# -----------------------------------------------------------------------------
def C1():
    hdr("C1  Maxwell-Juttner manifold: plain determinant (50 digits), several D, zeta, alpha")
    rows = []
    for D in (1, 2, 3, 4):
        for zeta in (0.0, 0.5, 3.0):
            for al in (-2.0, 0.0, 1.0, 10.0):       # alpha<0 allowed for the classical gas
                det, big, _ = tg.plain_det_scale_mp(zeta, al, tg.Stat("bg", 0), D=D, dps=50)
                ratio = float(abs(det) / big)
                rows.append(dict(D=D, zeta=zeta, alpha=al, det_over_term=ratio))
    worst = max(r["det_over_term"] for r in rows)
    print("max |det| / max|term| over %d points (D=1..4) = %.2e  (exactly 0 analytically)" % (len(rows), worst))
    RES["C1"] = dict(points=len(rows), max_det_over_term=worst)


# -----------------------------------------------------------------------------
def C2_N2_N3():
    hdr("C2/N2  dilute-limit constant c1(zeta) (D=3):  Rhat -> -a c1;  R_0 = -2 Xi B_2")
    zetas = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 100.0]
    old = {0.0: 0.0625, 0.1: 0.0619, 0.5: 0.0535, 1.0: 0.0419, 2.0: 0.0269, 4.0: 0.0141, 8.0: 0.0062}
    table = []
    print("%7s %20s %20s %13s %13s %10s %12s %12s %12s %8s" %
          ("zeta", "c1 (closed form)", "c1 (direct, a=-1)", "Delta_old", "Xi", "B2red(a=1)",
           "UR 1/16(1-z^2)", "NR lead", "NR 2-term", "old"))
    with mp.workdps(40):
        for z in zetas:
            c1 = tg.c1_closed(z, dps=40)
            # direct limit from the full curvature (alpha = 80: ztilde = 1.8e-35)
            Rdir = -tg.curvature_mp(z, 80, tg.Stat("bg", -1), dps=40) * (-1)
            RdirF = tg.curvature_mp(z, 80, tg.Stat("bg", 1), dps=40)
            if z > 0:
                c1o, Delta, G = tg.c1_old_determinant(z, dps=40)
            else:
                c1o, Delta, G = mp.mpf(1) / 16, mp.mpf(-9), mp.mpf(12)
            Xi = tg.Xi_dilute(z, dps=40)
            B2 = tg.B2_reduced(z, 1, dps=40)
            ur = (1 - z * z) / 16
            nr = 5 / (16 * mp.sqrt(mp.pi)) * mp.power(z, -1.5) if z > 0 else mp.inf
            nr2 = nr * (1 - mp.mpf(25) / (16 * z)) if z > 0 else mp.inf
            print("%7g %20s %20s %13s %13s %10s %12.6g %12.6g %12.6g %8s" %
                  (z, mp.nstr(c1, 15), mp.nstr(Rdir, 15), mp.nstr(Delta, 6), mp.nstr(Xi, 10), mp.nstr(B2, 7),
                   float(ur), float(nr), float(nr2), old.get(z, "")))
            table.append(dict(zeta=z, c1=float(c1), c1_str=mp.nstr(c1, 20), c1_direct_BE=float(Rdir),
                              c1_direct_FD=float(-RdirF), c1_old_formula=float(c1o), Delta_old=float(Delta),
                              G_old=float(G), Xi=float(Xi), B2_reduced_a1=float(B2),
                              R0_over_B2=float(-2 * Xi), UR_approx=float(ur),
                              NR_lead=float(nr) if z > 0 else None, NR_2term=float(nr2) if z > 0 else None,
                              old_table=old.get(z)))
    RES["C2_table"] = table
    # monotonicity / positivity on a fine log grid
    zs = np.logspace(-4, 4, 801)
    with mp.workdps(30):
        cs = [float(tg.c1_closed(z, dps=30)) for z in zs]
    dif = np.diff(cs)
    print("fine grid zeta in [1e-4, 1e4] (801 pts): min c1 = %.3e (>0: %s); strictly decreasing: %s" %
          (min(cs), min(cs) > 0, bool(np.all(dif < 0))))
    RES["C2_monotone"] = dict(min_c1=min(cs), positive=bool(min(cs) > 0), decreasing=bool(np.all(dif < 0)))
    # NR check of the 1/zeta coefficient
    with mp.workdps(40):
        for z in (1e3, 1e4):
            ratio = tg.c1_closed(z, dps=40) * mp.power(z, 1.5) * 16 * mp.sqrt(mp.pi) / 5
            pred = 1 - mp.mpf(25) / (16 * z) - mp.mpf(1503) / (512 * z * z)
            print("NR check zeta=%g: c1 zeta^{3/2} 16 sqrt(pi)/5 = %s ; 1-25/(16z)-1503/(512 z^2) = %s" %
                  (z, mp.nstr(ratio, 14), mp.nstr(pred, 14)))
    # General formula check on exact A, B (D=3, zeta=1.7, a=-1): series coefficients of phi
    with mp.workdps(40):
        zeta = mp.mpf("1.7")
        A = lambda b: zeta ** 2 * mp.besselk(2, b * zeta) / b              # m = zeta, eps-weighted
        B = lambda b: -(-1) / 2 * A(2 * b)
        R0 = tg.R0_from_virial(A, B)(mp.mpf(1))
        print("general formula R0(A,B) at zeta=1.7, a=-1: %s ; closed form +c1 = %s" %
              (mp.nstr(R0, 15), mp.nstr(tg.c1_closed(zeta, dps=40), 15)))
        RES["N2_general_formula_check"] = [float(R0), float(tg.c1_closed(zeta, dps=40))]
    # other dimensions
    hdr("C2 (D = 2, 4): closed form vs direct limit")
    dd = []
    with mp.workdps(40):
        for D in (2, 4):
            for z in (0.0, 0.5, 2.0):
                c1 = tg.c1_closed(z, D=D, dps=40)
                Rdir = tg.curvature_mp(z, 80, tg.Stat("bg", -1), D=D, dps=40)
                ur = (mp.mpf(D) + 1) / 4 * mp.power(2, -(3 * mp.mpf(D) - 1) / 2) / (
                    mp.gamma(mp.mpf(D) / 2) * mp.power(2, (mp.mpf(D) - 1) / 2) / mp.sqrt(mp.pi)
                    * mp.gamma((mp.mpf(D) + 1) / 2))
                print("D=%d zeta=%g: c1 closed %s, direct %s ; UR value %s ; Xi=%s" %
                      (D, z, mp.nstr(c1, 14), mp.nstr(Rdir, 14), mp.nstr(ur, 10),
                       mp.nstr(tg.Xi_dilute(z, D=D), 10)))
                dd.append(dict(D=D, zeta=z, c1=float(c1), direct=float(Rdir), UR=float(ur)))
    RES["C2_otherD"] = dd


# -----------------------------------------------------------------------------
def N3_polylog():
    hdr("N3  NR polylog curvature  R_NR = -5 (lambda^3/g)[2 f52 f12^2 - f52 f32 f-12 - f32^2 f12]/[5 f52 f12 - 3 f32^2]^2")
    out = []
    with mp.workdps(40):
        for a in (1, -1):
            lim = tg.R_powerlaw_quantum(mp.mpf("1e-12"), a, mp.mpf(3) / 2, c=1)
            print("a=%+d: R_NR(z=1e-12) (lambda^3/g units) = %s ; -a 5/2^{9/2} = %s" %
                  (a, mp.nstr(lim, 15), mp.nstr(-a * 5 / mp.power(2, 4.5), 15)))
            for zeta in (30.0, 100.0, 1000.0):
                for zt in (1e-3, 0.3, 0.9):
                    al = -mp.log(zt)
                    Rrel = tg.curvature_mp(zeta, al, tg.Stat("bg", a), dps=30)
                    # reduced NR normalisation: phihat = (2m/beta)^{3/2} (sqrt(pi)/4) f_{5/2}
                    c = mp.power(2 * mp.mpf(zeta), 1.5) * mp.sqrt(mp.pi) / 4
                    Rnr = tg.R_powerlaw_quantum(zt, a, mp.mpf(3) / 2, c=c)
                    print("   zeta=%6g ztilde=%5g: R_rel=%s  R_NR=%s  rel.diff=%.3e  (x zeta: %.3f)" %
                          (zeta, zt, mp.nstr(Rrel, 12), mp.nstr(Rnr, 12), float(Rrel / Rnr - 1),
                           float((Rrel / Rnr - 1) * zeta)))
                    out.append(dict(a=a, zeta=zeta, ztilde=zt, R_rel=float(Rrel), R_NR=float(Rnr)))
    RES["N3_polylog"] = out


# -----------------------------------------------------------------------------
def C3_sign_grid():
    hdr("C3  sign rule on a dense grid (fast path; mp spot checks every 25th point)")
    al = np.concatenate([np.logspace(-10, -1, 300), np.linspace(0.1, 27.7, 300)[1:]])   # ztilde 1-1e-10 ... 1e-12
    zetas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    summary = []
    curves = {"alpha": al}
    for z in zetas:
        RF = np.array([tg.curvature_np(z, x, tg.Stat("bg", 1)) for x in al])
        RB = np.array([tg.curvature_np(z, x, tg.Stat("bg", -1)) for x in al])
        curves["RFD_%g" % z] = RF
        curves["RBE_%g" % z] = RB
        spots = []
        for i in range(0, len(al), 25):
            for a, arr in ((1, RF), (-1, RB)):
                Rm = float(tg.curvature_mp(z, al[i], tg.Stat("bg", a), dps=30))
                spots.append(abs(arr[i] / Rm - 1))
        mono_F = bool(np.all(np.diff(RF[::-1]) > 0))      # as ztilde increases (alpha decreases)
        mono_B = bool(np.all(np.diff(RB[::-1]) > 0))
        s = dict(zeta=z, max_RFD=float(RF.max()), min_RBE=float(RB.min()), FD_all_negative=bool(np.all(RF < 0)),
                 BE_all_positive=bool(np.all(RB > 0)), FD_monotone_increasing_in_ztilde=mono_F,
                 BE_monotone_increasing_in_ztilde=mono_B, max_spot_relerr=float(max(spots)))
        summary.append(s)
        print("zeta=%5g: max R_FD = %.6e (all<0: %s) ; min R_BE = %.6e (all>0: %s) ; monotone F/B: %s/%s ; "
              "spot fast/mp max rel err %.1e" % (z, s["max_RFD"], s["FD_all_negative"], s["min_RBE"],
                                                 s["BE_all_positive"], mono_F, mono_B, s["max_spot_relerr"]))
    RES["C3"] = summary
    # massless too
    RF0 = np.array([tg.curvature_np(0.0, x, tg.Stat("bg", 1)) for x in al])
    RB0 = np.array([tg.curvature_np(0.0, x, tg.Stat("bg", -1)) for x in al])
    curves["RFD_0"], curves["RBE_0"] = RF0, RB0
    print("zeta=0 (massless): max R_FD = %.6e, min R_BE = %.6e" % (RF0.max(), RB0.min()))
    np.savez(os.path.join(DATA, "bg_curves.npz"), **curves)
    # degenerate Fermi gas (ztilde > 1) -- outside the claim, recorded for completeness
    deg = []
    for z in (0.0, 1.0, 10.0):
        vals = [float(tg.curvature_mp(z, -x, tg.Stat("bg", 1), dps=30)) for x in (1.0, 5.0, 20.0, 50.0)]
        deg.append(dict(zeta=z, alpha=[-1, -5, -20, -50], R=vals))
        print("degenerate FD (mu > m), zeta=%g, alpha=-1,-5,-20,-50: %s" % (z, ["%.4e" % v for v in vals]))
    RES["C3_degenerate_FD"] = deg


# -----------------------------------------------------------------------------
def C4_BE_divergence():
    hdr("C4  Bose divergence at ztilde -> 1:  R ~ C_BE(zeta) alpha^{-1/2}")
    res = []
    with mp.workdps(45):
        for z in (0.01, 0.1, 1.0, 10.0, 30.0):
            mo = tg.moments_mp(1, 0, z, tg.Stat("bg", -1), red="none", dps=40)
            P, Q = mo["row1"][0], mo["row1"][1]
            A3, B3 = mo["row2"][0], mo["row2"][1]
            CBE = (P * B3 - Q * A3) / P ** 2 * mp.power(2 * mp.mpf(z), -1.5) / mp.pi
            CNR = 2 / (5 * mp.pi) * mp.power(2 * mp.mpf(z), -1.5) * mp.zeta(1.5) / mp.zeta(2.5)
            alphas = [mp.mpf(10) ** (-k) for k in range(2, 15)]
            Rs = [tg.curvature_mp(z, a_, tg.Stat("bg", -1), dps=40) for a_ in alphas]
            prods = [R * mp.sqrt(a_) for R, a_ in zip(Rs, alphas)]
            # fit R = C a^{-1/2} + C0 + C1 a^{1/2} on the three smallest alphas
            M = mp.matrix([[mp.power(a_, -0.5), 1, mp.sqrt(a_)] for a_ in alphas[-3:]])
            coef = mp.lu_solve(M, mp.matrix(Rs[-3:]))
            # local exponent d ln R / d ln alpha between the two smallest alphas
            nu_loc = -(mp.log(Rs[-1]) - mp.log(Rs[-2])) / (mp.log(alphas[-1]) - mp.log(alphas[-2]))
            print("zeta=%5g: C_BE(analytic)=%s  fit C=%s C0=%s ; NR-limit formula %s ; local exponent %s" %
                  (z, mp.nstr(CBE, 12), mp.nstr(coef[0], 12), mp.nstr(coef[1], 6), mp.nstr(CNR, 8),
                   mp.nstr(nu_loc, 8)))
            print("        R sqrt(alpha) at alpha=1e-2..1e-14: " + " ".join(mp.nstr(p_, 6) for p_ in prods))
            res.append(dict(zeta=z, C_BE=float(CBE), fit_C=float(coef[0]), fit_C0=float(coef[1]),
                            C_NRformula=float(CNR), local_exponent=float(nu_loc),
                            R_sqrt_alpha=[float(p_) for p_ in prods], alphas=[float(a_) for a_ in alphas]))
        # physical form of the NR coefficient: R ~ (2/(5 pi)) (2 zeta)^{-3/2} zeta(3/2)/zeta(5/2) alpha^{-1/2}
        # massless: logarithmic divergence
        Li = [mp.polylog(k, 1) for k in (2, 3, 4)]
        slope = Li[2] * Li[1] / (4 * Li[2] * Li[0] - 3 * Li[1] ** 2) ** 2
        alphas = [mp.mpf(10) ** (-k) for k in range(2, 15)]
        Rs = [tg.curvature_mp(0, a_, tg.Stat("bg", -1), dps=40) for a_ in alphas]
        slopes = [(Rs[i + 1] - Rs[i]) / mp.log(10) for i in range(len(Rs) - 1)]
        print("zeta=0: R / ln(1/alpha) slope (numerical, successive decades): " +
              " ".join(mp.nstr(s, 8) for s in slopes) + " ; analytic Li4 Li3/(4 Li4 Li2 - 3 Li3^2)^2 = %s"
              % mp.nstr(slope, 10))
        res.append(dict(zeta=0, log_slope_analytic=float(slope), log_slopes=[float(s) for s in slopes]))
    RES["C4"] = res


# -----------------------------------------------------------------------------
def C10_virial():
    hdr("C10  virial: P/(nT) - 1 = a z 2^{-(D+1)/2} K_nu(2 zeta)/K_nu(zeta) + O(z^2)")
    res = []
    for D in (3,):
        for z in (0.0, 0.3, 1.0, 5.0, 30.0):
            for a in (1, -1):
                with mp.workdps(40):
                    nu = mp.mpf(D + 1) / 2
                    # coefficient of z (fugacity e^{mu/T}) and of ztilde
                    if z == 0:
                        cz = mp.mpf(a) * mp.power(2, -(D + 1))
                    else:
                        cz = a * mp.power(2, -nu) * mp.besselk(nu, 2 * z) / mp.besselk(nu, z)
                    czt = cz * mp.exp(z)
                    old = mp.mpf(a) / mp.power(2, mp.mpf(D) / 2 + 1)
                # numerical: two small ztilde values, eliminate O(ztilde^2)
                zts = [1e-6, 2e-6]
                vals = []
                for zt in zts:
                    ph, n, _ = tg.potential_np(z, -math.log(zt), tg.Stat("bg", a), D=D)
                    vals.append((ph / n - 1) / zt)
                num = 2 * vals[0] - vals[1]
                print("D=%d zeta=%5g a=%+d: [P/(nT)-1]/ztilde numeric %.10f | formula (ztilde) %.10f | "
                      "per z %.6f | old a/2^{D/2+1} %.6f" % (D, z, a, num, float(czt), float(cz), float(old)))
                res.append(dict(D=D, zeta=z, a=a, numeric_per_ztilde=num, formula_per_ztilde=float(czt),
                                formula_per_z=float(cz), old_formula=float(old)))
    RES["C10"] = res


def main():
    t0 = time.time()
    C1()
    C2_N2_N3()
    N3_polylog()
    C3_sign_grid()
    C4_BE_divergence()
    C10_virial()
    with open(os.path.join(DATA, "bg_results.json"), "w") as fh:
        json.dump(RES, fh, indent=1, default=str)
    print("\ntotal time %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
