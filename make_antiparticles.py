# -*- coding: utf-8 -*-
r"""
make_antiparticles.py -- numbers, checks and preview panel for the particle+antiparticle gas
============================================================================================

    python make_antiparticles.py

Outputs
    data/antiparticles.json                         every number (table, checks, limits, curves)
    data/antiparticles.log                          (redirect stdout)
    FIGDIR/fig2_antiparticles.{pdf,png}    (../manuscript/figures if it exists, else ./figures)

Conventions: thermogeom.py / antiparticles.py.  Rhat = (g T^3/2 pi^2) R, geometer's sign.
mu = 0 throughout except the sign-change scan of BE+- in mu (sec. 6).
Single species at mu = 0 means ztilde = e^{-zeta}, alpha = zeta.
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import mpmath as mp
import numpy as np

import antiparticles as ap
import thermogeom as tg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
_MS_FIGS = os.path.normpath(os.path.join(HERE, "..", "manuscript", "figures"))
FIGDIR = _MS_FIGS if os.path.isdir(_MS_FIGS) else os.path.join(HERE, "figures")
STYLE = os.path.join(HERE, "style.mplstyle")
NPROC = max(1, (os.cpu_count() or 2) - 2)
CM = 1.0 / 2.54
FIGW = 15.5 * CM
COL = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#AA3377', '#555555', '#DDAA33', '#33BBEE']

ST = {"FD": tg.Stat("bg", 1), "BE": tg.Stat("bg", -1), "MJ": tg.Stat("bg", 0)}
TABLE_ZETAS = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 30]
DPS = 50


def rel(x, y):
    """relative difference, absolute if the reference vanishes (flat single-species MJ)."""
    return abs(x - y) / abs(y) if y != 0 else abs(x - y)


def s(v, n=15):
    return mp.nstr(v, n)


# ---------------------------------------------------------------- mp workers
def _w_table(args):
    name, zeta = args
    st = ST[name]
    with mp.workdps(DPS + 10):
        Rpm = ap.curvature_pm(zeta, 0.0, st, path="mp", dps=DPS)
        Rpm40 = ap.curvature_pm(zeta, 0.0, st, path="mp", dps=40)
        Rsym = ap.curvature_mu0_symmetric_mp(zeta, st, dps=DPS)
        R1 = ap.curvature_pm(zeta, 0.0, st, path="mp", dps=DPS, antiparticles=False)
        R1tg = tg.curvature_mp(zeta, zeta, st, dps=DPS)          # thermogeom, reduced det
        out = dict(Rpm=s(Rpm, 20), Rsingle=s(R1, 20),
                   rel_dps40_vs_50=s(rel(Rpm40, Rpm), 3),
                   rel_symmetric_formula=s(rel(Rsym, Rpm), 3),
                   rel_single_vs_thermogeom=s(rel(R1, R1tg), 3))
        out["np_Rpm"] = ap.curvature_pm(zeta, 0.0, st)
        out["rel_np_vs_mp"] = float(rel(mp.mpf(out["np_Rpm"]), Rpm))
        R1np = tg.curvature_np(zeta, zeta, st)
        out["np_Rsingle"] = R1np
        out["rel_np_single_vs_mp"] = float(rel(mp.mpf(R1np), R1tg))
        if name == "MJ":
            out["rel_vs_closed"] = s(abs(Rpm / ap.R_MJ_pm(zeta, 0, dps=DPS) - 1), 3)
    return name, zeta, out


def _w_mu(args):
    """BE+- / FD+- at mu != 0: np vs mp spot check."""
    name, zeta, x = args
    st = ST[name]
    with mp.workdps(DPS + 10):
        Rm = ap.curvature_pm(zeta, x, st, path="mp", dps=DPS)
        Rn = ap.curvature_pm(zeta, x, st)
        return name, zeta, x, s(Rm, 16), float(abs(mp.mpf(Rn) / Rm - 1))


def _w_small_zeta(args):
    name, zeta = args
    with mp.workdps(DPS + 10):
        return name, zeta, s(ap.curvature_pm(zeta, 0.0, ST[name], path="mp", dps=30), 16)


def _w_root_mu(zeta):
    """x* = mu*/T where Rhat_BE+-(zeta, x) changes sign (0 < x* < zeta)."""
    st = ST["BE"]
    xs = np.linspace(0.0, zeta * (1 - 1e-6), 400)
    R = [ap.curvature_pm(zeta, float(x), st) for x in xs]
    idx = tg.find_sign_changes(xs, R)
    roots = []
    from scipy.optimize import brentq
    for i in idx:
        roots.append(brentq(lambda x: ap.curvature_pm(zeta, x, st), xs[i], xs[i + 1], xtol=1e-14))
    # refine the first root on the mp path
    ref = None
    if roots:
        with mp.workdps(30):
            f = lambda x: ap.curvature_pm(zeta, x, st, path="mp", dps=25)
            ref = float(mp.findroot(f, (mp.mpf(roots[0]) * (1 - 1e-6), mp.mpf(roots[0]) * (1 + 1e-6)),
                                    solver="anderson"))
    return zeta, roots, ref, float(R[0])


def pmap(fn, items):
    with ProcessPoolExecutor(NPROC) as ex:
        return list(ex.map(fn, items, chunksize=1))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    J = {"conventions": "Rhat = (g T^3/2pi^2) R, geometer's sign (2-sphere > 0); mu = 0 unless stated; "
                        "single species at mu = 0 has ztilde = e^-zeta; 'pm' = with antiparticles."}

    # ---------------------------------------------------------- 1. table + accuracy
    print("== 1. table at mu = 0 (mp plain determinant, %d digits) ==" % DPS)
    res = pmap(_w_table, [(n, z) for n in ("FD", "BE", "MJ") for z in TABLE_ZETAS])
    tab = {n: {} for n in ("FD", "BE", "MJ")}
    for n, z, o in res:
        tab[n]["%g" % z] = o
    for z in TABLE_ZETAS:
        k = "%g" % z
        with mp.workdps(40):
            RMJc = ap.R_MJ_pm(z, 0)
            D = ap.Delta_dilute(z)

        tab["MJ"][k]["closed"] = s(RMJc, 20)
        tab["MJ"][k]["NR_series"] = ap.R_MJ_pm_NR(z) if z >= 1 else None
        tab["FD"][k]["Delta_closed"] = s(D, 16)
        with mp.workdps(40):
            dFD = mp.mpf(tab["FD"][k]["Rpm"]) - RMJc
            dBE = mp.mpf(tab["BE"][k]["Rpm"]) - RMJc
        tab["FD"][k]["Rpm_minus_RMJ"] = s(dFD, 16)
        tab["BE"][k]["Rpm_minus_RMJ"] = s(dBE, 16)
        tab["FD"][k]["r0"] = s(tg.c1_closed(z, dps=30), 16)
    J["table_mu0"] = tab
    print("%6s | %22s %22s | %22s %22s | %22s" % ("zeta", "FD+-", "FD single", "BE+-", "BE single", "MJ+- (closed)"))
    for z in TABLE_ZETAS:
        k = "%g" % z
        print("%6g | %22s %22s | %22s %22s | %22s" % (
            z, tab["FD"][k]["Rpm"][:20], tab["FD"][k]["Rsingle"][:20], tab["BE"][k]["Rpm"][:20],
            tab["BE"][k]["Rsingle"][:20], tab["MJ"][k]["closed"][:20]))
    print("\naccuracy (relative):  dps40-vs-50 | sym.formula | single vs thermogeom | np vs mp (pm) | np vs mp (single)")
    for n in ("FD", "BE", "MJ"):
        for z in TABLE_ZETAS:
            o = tab[n]["%g" % z]
            print("  %s %6g  %s | %s | %s | %.2e | %.2e %s" % (
                n, z, o["rel_dps40_vs_50"], o["rel_symmetric_formula"], o["rel_single_vs_thermogeom"],
                o["rel_np_vs_mp"], o["rel_np_single_vs_mp"],
                ("| closed %s" % o["rel_vs_closed"]) if n == "MJ" else ""))
    print("\ndilute decomposition  Rpm = R_MJpm + a Delta + O(e^-zeta):")
    print("%6s %20s %20s %20s %20s %16s" % ("zeta", "FD:Rpm-RMJ", "BE:Rpm-RMJ", "Delta(closed)", "Delta(NR)", "r0 (single)"))
    for z in TABLE_ZETAS:
        k = "%g" % z
        print("%6g %20s %20s %20s %20.12g %16s" % (z, tab["FD"][k]["Rpm_minus_RMJ"], tab["BE"][k]["Rpm_minus_RMJ"],
                                                   tab["FD"][k]["Delta_closed"], ap.Delta_NR(z), tab["FD"][k]["r0"]))

    # ---------------------------------------------------------- 2. mu != 0 accuracy spot checks
    print("\n== 2. mu != 0: np vs mp ==")
    items = [(n, z, x) for n in ("FD", "BE") for z in (0.5, 2, 10) for x in (0.3 * z, 0.9 * z)]
    items += [("FD", 1.0, -3.0), ("FD", 1.0, 5.0)]
    mu_chk = []
    for n, z, x, Rm, rel in pmap(_w_mu, items):
        print("  %s zeta=%g x=%g  R=%s  np/mp-1=%.2e" % (n, z, x, Rm, rel))
        mu_chk.append(dict(stat=n, zeta=z, x=x, R=Rm, rel_np_mp=rel))
    J["mu_nonzero_checks"] = mu_chk

    # ---------------------------------------------------------- 3. massless limits
    print("\n== 3. massless limit, mu = 0 ==")
    exact = {n: ap.R_massless_mu0(a) for n, a in (("FD", 1), ("BE", -1), ("MJ", 0))}
    small = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    sm = pmap(_w_small_zeta, [(n, z) for n in ("FD", "BE", "MJ") for z in small])
    lim = {n: {"exact_m0": exact[n], "approach": {}} for n in exact}
    for n, z, R in sm:
        lim[n]["approach"]["%g" % z] = R
    for n in ("FD", "BE", "MJ"):
        print("  %s exact(m=0) = %.15f" % (n, exact[n]))
        prev = None
        for z in small:
            R = float(lim[n]["approach"]["%g" % z])
            dR = R - exact[n]
            slope = ("  local exponent %.4f" % (math.log(abs(dR / prev[1])) / math.log(z / prev[0]))) if (prev and dR != 0 and prev[1] != 0) else ""
            print("     zeta=%g  R=%.15f  R-R0=%.4e%s" % (z, R, dR, slope))
            prev = (z, dR)
    # small-zeta approach.  BE: Rhat = -45/(8 pi^4) + 45 zeta/(32 pi^5) + ...  (infrared: delta chi = -pi zeta/2,
    # delta d_beta chi = +pi zeta per species).  FD: Rhat - R0 = c_FD zeta^2 (numerical coefficient).
    zz = 1e-5
    slopeBE = (float(lim["BE"]["approach"]["%g" % zz]) - exact["BE"]) / zz
    lim["BE"]["linear_slope_numeric"] = slopeBE
    lim["BE"]["linear_slope_closed"] = 45 / (32 * math.pi ** 5)
    lim["FD"]["quadratic_coef_numeric"] = (float(lim["FD"]["approach"]["1e-05"]) - exact["FD"]) / 1e-10
    print("  BE+- slope: numeric %.8e  closed 45/(32 pi^5) = %.8e" % (slopeBE, 45 / (32 * math.pi ** 5)))
    print("  FD+- zeta^2 coefficient (numeric, from zeta=1e-3):",
          (float(lim["FD"]["approach"]["0.001"]) - exact["FD"]) / 1e-6)
    # single-species massless mu=0 for comparison: FD at ztilde=1 (polylog), BE diverges
    R1FD0 = tg.R_powerlaw_quantum(1, 1, 3, c=2)
    lim["FD"]["single_m0"] = s(R1FD0, 16)
    lim["BE"]["single_m0"] = "+inf (logarithmic)"
    print("  single FD massless mu=0:", s(R1FD0, 16))
    # single-species BE as zeta -> 0 at mu = 0 (alpha = zeta)
    lim["BE"]["single_approach"] = {"%g" % z: tg.curvature_np(z, z, ST["BE"]) for z in small[:5]}
    print("  single BE (mu=0):", lim["BE"]["single_approach"])
    # FD+- massless closed form vs mp at x != 0
    chk = []
    with mp.workdps(40):
        for x in (0.5, 2.0, -4.0):
            Rm = ap.curvature_pm(1e-8, x, ST["FD"], path="mp", dps=30)
            chk.append((x, s(Rm, 15), ap.R_massless_FD_pm(x)))
            print("  FD+- zeta=1e-8 x=%g: mp %s  closed(m=0) %.15f" % (x, s(Rm, 15), ap.R_massless_FD_pm(x)))
    lim["FD"]["closed_form_x_checks"] = chk
    J["massless"] = lim

    # ---------------------------------------------------------- 4. curves
    print("\n== 4. curves (fast path) ==")
    zg = np.logspace(-2, math.log10(30), 241)
    cur = {"zeta": zg.tolist()}
    for n in ("FD", "BE"):
        cur[n + "_pm"] = [ap.curvature_pm(float(z), 0.0, ST[n]) for z in zg]
        cur[n + "_single"] = [tg.curvature_np(float(z), float(z), ST[n]) for z in zg]
    cur["MJ_pm"] = [float(ap.R_MJ_pm(float(z), 0, dps=20)) for z in zg]
    # P/T = phihat (reduced), for the scale-free combination Rhat * phihat_tot = R P/T
    for n in ("FD", "BE"):
        cur[n + "_phihat_single"] = [tg.potential_np(float(z), float(z), ST[n])[0] for z in zg]
    cur["MJ_phihat_single"] = [float(ap.A_total(1, mp.mpf(z))) for z in zg]
    for n in ("FD", "BE"):
        print("  %s+-: all negative on grid: %s; sign changes at indices %s" % (n, all(r < 0 for r in cur[n + "_pm"]), tg.find_sign_changes(zg, cur[n + "_pm"])))

    print("  max FD+- = %.6e   max BE+- = %.6e   (closest to zero)" % (max(cur["FD_pm"]), max(cur["BE_pm"])))
    # extremum of BE+- / FD+- in zeta (|R| minimum)
    for n in ("FD", "BE", "MJ"):
        arr = np.array(cur[n + "_pm"])
        i = int(np.argmax(arr))
        print("  %s+-: least negative at zeta ~ %.3f, R = %.6f" % (n, zg[i], arr[i]))
        J.setdefault("extrema", {})[n] = dict(zeta=float(zg[i]), R=float(arr[i]))
    J["curves_mu0"] = cur

    # ---------------------------------------------------------- 5. scale-free R*P/T
    # (computed in the figure section; massless universal value -1/4)

    # ---------------------------------------------------------- 6. BE+- sign change in mu
    print("\n== 6. BE+- : sign change as |mu| -> m ==")
    sc = pmap(_w_root_mu, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    J["BE_pm_sign_change_in_mu"] = []
    for z, roots, ref, R0 in sc:
        print("  zeta=%g  R(mu=0)=%.6e  roots x*=mu*/T: %s  mp-refined %s  -> mu*/m = %s" % (
            z, R0, ["%.10f" % r for r in roots], ref, None if ref is None else "%.8f" % (ref / z)))
        J["BE_pm_sign_change_in_mu"].append(dict(zeta=z, roots_np=roots, root_mp=ref,
                                                 mu_star_over_m=None if ref is None else ref / z))
    with open(os.path.join(DATA, "antiparticles.json"), "w") as f:
        json.dump(J, f, indent=1)
    print("\nwrote data/antiparticles.json")
    make_figure(J)
    print("total %.0f s" % (time.time() - t0))


# ---------------------------------------------------------------- figure
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(STYLE)
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "lines.linewidth": 1.4, "legend.handlelength": 1.8,
        "figure.constrained_layout.use": True, "savefig.bbox": "standard",
        "figure.constrained_layout.w_pad": 0.03, "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.wspace": 0.04,
    })
    return plt


def make_figure(J):
    from matplotlib.lines import Line2D
    plt = setup_mpl()
    c = J["curves_mu0"]
    z = np.array(c["zeta"])
    fig, axs = plt.subplots(1, 2, figsize=(FIGW * 2 / 3 + 0.6, 5.5 * CM))
    # ---- (a) |Rhat|, log-log
    ax = axs[0]
    ax.plot(z, -np.array(c["FD_pm"]), color=COL[0], ls="-")
    ax.plot(z, -np.array(c["FD_single"]), color=COL[0], ls="--")
    ax.plot(z, -np.array(c["BE_pm"]), color=COL[1], ls="-")
    ax.plot(z, np.array(c["BE_single"]), color=COL[1], ls="--")
    ax.plot(z, -np.array(c["MJ_pm"]), color="k", ls=":", lw=1.1)
    zn = z[z >= 4]
    ax.plot(zn, 0.75 * math.sqrt(2 / math.pi) * zn ** -3.5 * np.exp(zn), color=COL[5], lw=0.7,
            ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 30)
    ax.set_ylim(3e-4, 3e8)
    ax.set_yticks([1e-3, 1e0, 1e3, 1e6])
    ax.text(9.0, 2e5, r"$\propto\!\zeta^{-7/2}e^{\zeta}$", fontsize=8, color=COL[5], ha="right")
    ax.set_xlabel(r"$\zeta=m/T$")
    ax.set_ylabel(r"$|\hat R|$ at $\mu=0$")
    h = [Line2D([], [], color=COL[0], label="FD"), Line2D([], [], color=COL[1], label="BE"),
         Line2D([], [], color="k", ls=":", lw=1.1, label=r"MJ$^{\pm}$"),
         Line2D([], [], color="0.3", ls="-", label="with antiparticles"),
         Line2D([], [], color="0.3", ls="--", label="one species")]
    ax.legend(handles=h, loc="upper left", ncol=1)
    ax.set_title("(a)", loc="left", fontsize=9, pad=3)
    # ---- (b) R P/T = Rhat * phihat_tot (dimensionless, scale-free)
    ax = axs[1]
    for n, col in (("FD", COL[0]), ("BE", COL[1])):
        ph = np.array(c[n + "_phihat_single"])
        ax.plot(z, np.array(c[n + "_pm"]) * 2 * ph, color=col, ls="-")
        ax.plot(z, np.array(c[n + "_single"]) * ph, color=col, ls="--")
    ax.plot(z, np.array(c["MJ_pm"]) * 2 * np.array(c["MJ_phihat_single"]), color="k", ls=":", lw=1.1)
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.axhline(-0.25, color="0.6", lw=0.6, ls=(0, (1, 2)), zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 30)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel(r"$\zeta=m/T$")
    ax.set_ylabel(r"$R\,P/T$ at $\mu=0$")
    ax.text(0.012, -0.243, r"$-1/4$", fontsize=8, color="0.35", va="bottom")
    ax.set_title("(b)", loc="left", fontsize=9, pad=3)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(os.path.join(FIGDIR, "fig2_antiparticles.%s" % ext), **kw)
    plt.close(fig)
    print("wrote fig2_antiparticles.pdf/.png")


if __name__ == "__main__":
    main()
