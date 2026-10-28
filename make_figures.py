# -*- coding: utf-8 -*-
r"""
make_figures.py -- Figs. 1, 3 and 4 of the NPB manuscript (Ruppeiner geometry of relativistic
ideal gases and their Tsallis / Kaniadakis deformations).

    python make_figures.py              # use cached curves where present
    python make_figures.py --recompute  # regenerate every curve from thermogeom.py

Outputs (FIGDIR = ../manuscript/figures if that folder exists, otherwise ./figures)
    FIGDIR/fig1_bg.{pdf,png}, fig4_kaniadakis.{pdf,png}, fig3_limits.{pdf,png}
    data/fig_cache_fig{1,2,3}.npz       (curve cache)
    data/figures_numbers.json           (every number quoted in the text / captions)

Conventions (identical to thermogeom.py):
    Rhat = (g T^3 / 2 pi^2) R, geometer's sign (two-sphere > 0): FD < 0, BE > 0.
    zeta = m/T, ztilde = e^{(mu-m)/T}, alpha = -ln ztilde = (m - mu)/T, gamma = -mu/T = alpha - zeta.
    r0(zeta)      = thermogeom.c1_closed        (Rhat -> -a r0 as ztilde -> 0)
    Gamma_B(zeta)                               (Rhat_BE ~ Gamma_B alpha^{-1/2}, alpha -> 0)
    Upsilon(zeta) = thermogeom.Y_kappa,  A(zeta) = thermogeom.A_reduced(kinetic=True)
    C(q)  : Rhat -> -a C(q) gamma'^{-4},           gamma' = gamma + 1/(q-1)   (beta = 1)
    K(k)  : Rhat -> -K(k) gamma^{1/k - 6}          (classical Kaniadakis, beta = 1)
Numerics: BG curves use the fast float64 path (validated to ~1e-14); every deformed curve and
every crossover is computed on the mpmath path (dps = 20 for curves, 30 for roots).
"""
from __future__ import annotations

import argparse
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
_MS_FIGS = os.path.normpath(os.path.join(HERE, "..", "manuscript", "figures"))
FIGDIR = _MS_FIGS if os.path.isdir(_MS_FIGS) else os.path.join(HERE, "figures")
STYLE = os.path.join(HERE, "style.mplstyle")
NPROC = max(1, (os.cpu_count() or 2) - 2)
CM = 1.0 / 2.54
FIGW = 15.5 * CM

# reference values from an independent verification run (sanity checks)
REPORT = {
    "r0(1)": 0.0418904471474299,
    "r0(0)": 0.0625,
    "Gamma_B(0.1)": 1.06317202286, "Gamma_B(1)": 0.0526957324486,
    "Gamma_B(10)": 0.00267165182344, "Gamma_B(30)": 0.000530610050594,
    "zstar(0.05,1)": 0.023817, "zstar(0.1,1)": 0.091950, "zstar(0.15,1)": 0.205874,
    "zstar(0.2,1)": 0.372323, "zstar(0.24,1)": 0.548679,
    "zstar(0.05,0.1)": 0.034492, "zstar(0.1,0.1)": 0.133145, "zstar(0.15,0.1)": 0.294728,
    "zstar(0.2,0.1)": 0.516206, "zstar(0.24,0.1)": 0.720184,
    "zstar(0.05,10)": 0.015269, "zstar(0.1,10)": 0.057286, "zstar(0.15,10)": 0.125919,
    "zstar(0.2,10)": 0.231321, "zstar(0.24,10)": 0.373953,
    "lim zstar/k2 (0.1)": 14.6276, "lim zstar/k2 (1)": 10.1463, "lim zstar/k2 (10)": 6.6461,
    "C(1.1)": 147.45510836, "C(1.2)": 16.0 / 21.0,
    "K(0.24)": 0.01415027327, "K(0.2)": 0.08192, "K(1/6)": 0.07054673721, "K(0.1)": 0.0001970548364,
}


# =============================================================================
# constants
# =============================================================================
def r0(zeta):
    return float(tg.c1_closed(zeta, dps=40))


def Gamma_B(zeta, dps=40):
    """Gamma_B: Rhat_BE ~ Gamma_B alpha^{-1/2} (alpha -> 0, zeta > 0)."""
    with mp.workdps(dps + 5):
        mo = tg.moments_mp(1, 0, zeta, tg.Stat("bg", -1), red="none", dps=dps)
        P, Q = mo["row1"][0], mo["row1"][1]
        A3, B3 = mo["row2"][0], mo["row2"][1]
        return float((P * B3 - Q * A3) / P ** 2 * mp.power(2 * mp.mpf(zeta), -1.5) / mp.pi)


def Upsilon(zeta):
    return float(tg.Y_kappa(zeta, dps=40))


def A_kin(zeta):
    return float(tg.A_reduced(zeta, kinetic=True, dps=40))


def zstar_lead_coef(zeta):
    """lim_{kappa->0} ztilde*/kappa^2 = Upsilon / (r0 A)."""
    return Upsilon(zeta) / (r0(zeta) * A_kin(zeta))


def C_tsallis(q):
    p = 1.0 / (q - 1.0)
    return (p - 1) ** 2 * (p - 2) * (p - 3) ** 2 * (p - 4) / (4 * (2 * p - 1) * (2 * p - 3))


def K_kaniadakis(kappa):
    P = mp.mpf(1) / mp.mpf(kappa)
    return float(mp.power(2 * mp.mpf(kappa), P) * (P - 1) * (P - 2) * (P - 3) ** 2 * (P - 4) / (P + 1))


# =============================================================================
# parallel mp workers
# =============================================================================
def _w_curv(args):
    kind, a, par, zeta, alpha, dps = args
    return float(tg.curvature_mp(zeta, alpha, tg.Stat(kind, a, par), dps=dps))


def _w_root(args):
    """Refine a crossover alpha* of the Kaniadakis Bose curvature on the mp path."""
    kappa, zeta, lo, hi, dps = args
    st = tg.Stat("k", -1, kappa)
    with mp.workdps(dps + 10):
        f = lambda x: tg.curvature_mp(zeta, x, st, dps=dps)
        r = mp.findroot(f, (mp.mpf(lo), mp.mpf(hi)), solver="illinois", tol=mp.mpf(10) ** -(dps - 6))
        return float(r)


def pmap(fn, items):
    with ProcessPoolExecutor(NPROC) as ex:
        return list(ex.map(fn, items, chunksize=4))


def bg_curve(zeta, a, alphas):
    st = tg.Stat("bg", a)
    return np.array([tg.curvature_np(zeta, float(x), st) for x in alphas])


def cache_path(name):
    return os.path.join(DATA, "fig_cache_%s.npz" % name)


def load_or(name, fn, recompute):
    path = cache_path(name)
    if not recompute and os.path.exists(path):
        print("  [cache] %s" % os.path.basename(path))
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    t0 = time.time()
    d = fn()
    np.savez(path, **d)
    print("  [computed] %s in %.1f s" % (os.path.basename(path), time.time() - t0))
    return d


# =============================================================================
# Fig. 1 data
# =============================================================================
F1_ZETAS_A = (0.0, 1.0, 10.0)
F1_ZETAS_C = (0.1, 1.0, 10.0, 30.0)


def compute_fig1():
    d = {}
    # (a) alpha grid: ztilde from 1e-6 to 1 - 1e-12 (dense near condensation)
    al = np.unique(np.concatenate([np.linspace(-math.log(1e-6), 0.5, 220),
                                   np.logspace(math.log10(0.5), -12, 160)]))[::-1]
    d["a_alpha"] = al
    for z in F1_ZETAS_A:
        for a in (1, -1):
            d["a_R_%g_%d" % (z, a)] = bg_curve(z, a, al)
    # (b) r0(zeta), closed form
    zs = np.logspace(-2, 3, 161)
    d["b_zeta"] = zs
    d["b_r0"] = np.array([r0(z) for z in zs])
    # direct-limit check of the closed form (fast path, ztilde = e^-60)
    zchk = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    d["b_zchk"] = zchk
    d["b_direct"] = np.array([-tg.curvature_np(z, 60.0, tg.Stat("bg", 1)) for z in zchk])
    # (c) approach to the BE divergence.  Plotted against alpha (not alpha/zeta): the curves
    # for zeta >= 1 collapse in alpha (NR polylog universality), not in alpha/zeta.
    for z in F1_ZETAS_C:
        alc = np.logspace(-11, 2, 196)
        d["c_alpha_%g" % z] = alc
        d["c_R_%g" % z] = bg_curve(z, -1, alc)
        d["c_Gamma_%g" % z] = np.array(Gamma_B(z))
    return d


# =============================================================================
# Fig. 2 data
# =============================================================================
F2_ZETA = 1.0
F2_KAPPAS = (0.05, 0.10, 0.15, 0.20)
F2B_ZETAS = (0.1, 1.0, 10.0)
F2B_TABLE_KAPPAS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.24)


def _crossovers(kappa, zeta):
    """Sign changes of the Kaniadakis-Bose Rhat(alpha) on a dense grid (fast path)."""
    st = tg.Stat("k", -1, kappa)
    al = np.logspace(-8, 3, 450)
    R = np.array([tg.curvature_np(zeta, float(x), st) for x in al])
    return [(al[i], al[i + 1]) for i in tg.find_sign_changes(al, R)]


def compute_fig2():
    d = {}
    # (a) curves at zeta = 1, mp path
    al = np.unique(np.concatenate([np.linspace(-math.log(1e-4), 0.3, 150),
                                   np.logspace(math.log10(0.3), -8, 60)]))[::-1]
    d["a_alpha"] = al
    d["a_R_bg"] = bg_curve(F2_ZETA, -1, al)
    items = [("k", -1, k, F2_ZETA, float(x), 20) for k in F2_KAPPAS for x in al]
    vals = np.array(pmap(_w_curv, items)).reshape(len(F2_KAPPAS), len(al))
    for k, v in zip(F2_KAPPAS, vals):
        d["a_R_%g" % k] = v
    # (b) crossovers: kappa grid + table values, three zetas
    kgrid = np.unique(np.round(np.concatenate([np.geomspace(0.005, 0.24, 30), F2B_TABLE_KAPPAS]), 6))
    d["b_kappa"] = kgrid
    jobs, meta = [], []
    nsc = np.zeros((len(F2B_ZETAS), len(kgrid)), dtype=int)
    for i, z in enumerate(F2B_ZETAS):
        for j, k in enumerate(kgrid):
            br = _crossovers(float(k), z)
            nsc[i, j] = len(br)
            lo, hi = br[0]
            jobs.append((float(k), z, float(lo), float(hi), 30))
            meta.append((i, j))
    roots = pmap(_w_root, jobs)
    alpha_star = np.zeros((len(F2B_ZETAS), len(kgrid)))
    for (i, j), r in zip(meta, roots):
        alpha_star[i, j] = r
    d["b_alpha_star"] = alpha_star
    d["b_nsignchanges"] = nsc
    d["b_lead"] = np.array([zstar_lead_coef(z) for z in F2B_ZETAS])
    return d


# =============================================================================
# Fig. 3 data
# =============================================================================
F3_ZETA = 1.0
F3_QS = (1.01, 1.05, 1.1, 1.2)
F3_KAPPAS = (0.10, 0.14, 1.0 / 6.0, 0.20, 0.24)     # 0.10 kept in cache for comparison only


def compute_fig3():
    d = {}
    ala = np.logspace(-1, 5, 121)
    d["a_alpha"] = ala
    d["a_R_bg"] = bg_curve(F3_ZETA, 1, ala)
    items = [("q", 1, q, F3_ZETA, float(x), 20) for q in F3_QS for x in ala]
    vals = np.array(pmap(_w_curv, items)).reshape(len(F3_QS), len(ala))
    for q, v in zip(F3_QS, vals):
        d["a_R_%g" % q] = v
    alb = np.logspace(0, 5, 101)
    d["b_alpha"] = alb
    items = [("k", 0, k, F3_ZETA, float(x), 20) for k in F3_KAPPAS for x in alb]
    vals = np.array(pmap(_w_curv, items)).reshape(len(F3_KAPPAS), len(alb))
    for k, v in zip(F3_KAPPAS, vals):
        d["b_R_%.6f" % k] = v
    return d


# =============================================================================
# plotting helpers
# =============================================================================
COL = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#AA3377', '#555555', '#DDAA33', '#33BBEE']


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


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontsize=9, pad=3)


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(os.path.join(FIGDIR, "%s.%s" % (name, ext)), **kw)
    print("  wrote %s.pdf/.png" % name)


# =============================================================================
# Fig. 1
# =============================================================================
def plot_fig1(plt, d, nums):
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(FIGW, 5.5 * CM))
    # ---- (a)
    ax = axs[0]
    zt = np.exp(-d["a_alpha"])
    for c, z in zip(COL, F1_ZETAS_A):
        rr = nums["r0"]["%g" % z]
        for a, ls in ((1, "-"), (-1, "--")):
            y = d["a_R_%g_%d" % (z, a)] / rr
            ax.plot(zt, np.where(y < 4, y, np.nan), ls=ls, color=c)
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1.0)
    ax.set_ylim(-1.25, 3.0)
    ax.set_xlabel(r"$\tilde z$")
    ax.set_ylabel(r"$\hat R/r_0(\zeta)$")
    h = [Line2D([], [], color=c, label=r"$\zeta=%g$" % z) for c, z in zip(COL, F1_ZETAS_A)]
    h += [Line2D([], [], color="k", ls="-", label="FD"), Line2D([], [], color="k", ls="--", label="BE")]
    ax.legend(handles=h, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
    panel_label(ax, "(a)")
    # ---- (b)
    ax = axs[1]
    zs, r = d["b_zeta"], d["b_r0"]
    ax.plot(zs, r, color=COL[0], label=r"$r_0(\zeta)$")
    zu = zs[zs <= 0.85]
    ax.plot(zu, (1 - zu ** 2) / 16, ls=":", color="k", lw=1.3, label=r"$(1-\zeta^2)/16$")
    zn = zs[zs >= 0.3]
    ax.plot(zn, 5 / (16 * math.sqrt(math.pi)) * zn ** -1.5, ls="--", color=COL[1], lw=1.1,
            label=r"$\frac{5}{16\sqrt{\pi}}\,\zeta^{-3/2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(1e-5, 0.2)
    ax.set_xlabel(r"$\zeta=m/T$")
    ax.set_ylabel(r"$r_0$")
    ax.legend(loc="lower left")
    panel_label(ax, "(b)")
    # ---- (c)
    ax = axs[2]
    for c, z in zip(COL, F1_ZETAS_C):
        al = d["c_alpha_%g" % z]
        y = np.sqrt(al) * d["c_R_%g" % z] / float(d["c_Gamma_%g" % z])
        ax.plot(al, y, color=c, label=r"$\zeta=%g$" % z)
    ax.axhline(1, color="k", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(1e-11, 1e2)
    ax.set_xticks([1e-10, 1e-7, 1e-4, 1e-1, 1e2])
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r"$\alpha=(m-\mu)/T$")
    ax.set_ylabel(r"$\sqrt{\alpha}\,\hat R_{\rm BE}/\Gamma_{\rm B}(\zeta)$")
    ax.legend(loc="upper left")
    panel_label(ax, "(c)")
    save(fig, "fig1_bg")
    plt.close(fig)


# =============================================================================
# Fig. 2
# =============================================================================
def plot_fig2(plt, d, nums):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.5 * CM))
    ax = axs[0]
    zt = np.exp(-d["a_alpha"])
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.plot(zt, d["a_R_bg"], color="k", ls="--", lw=1.2, label=r"$\kappa=0$")
    for c, k in zip(COL, F2_KAPPAS):
        ax.plot(zt, d["a_R_%g" % k], color=c, label=r"$\kappa=%.2f$" % k)
        zs = nums["kaniadakis_crossover"]["zeta=1"]["%.2f" % k]
        ax.plot([zs], [0], "o", ms=4.5, mfc="white", mec=c, mew=1.1, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("asinh", linear_width=0.02)
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(-20, 20)
    ticks = [-10, -1, -0.1, 0, 0.1, 1, 10]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: ("%g" % v).replace("-", "\u2212")))
    ax.set_xlabel(r"$\tilde z$")
    ax.set_ylabel(r"$\hat R_{\rm BE}$")
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
    panel_label(ax, "(a)")
    # (b)
    ax = axs[1]
    k = d["b_kappa"]
    for i, (c, z) in enumerate(zip(COL, F2B_ZETAS)):
        zst = np.exp(-d["b_alpha_star"][i])
        ax.plot(k, zst / k ** 2, color=c, label=r"$\zeta=%g$" % z)
        ax.plot([0.0042], [d["b_lead"][i]], marker="<", ms=5, mfc="white", mec=c, mew=1.1,
                ls="none", clip_on=False, zorder=5)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker="<", ms=5, mfc="white", mec="0.3", mew=1.1, ls="none"))
    l.append("leading order")
    ax.set_xscale("log")
    ax.set_xlim(0.004, 0.25)
    ax.set_ylim(1.5, 16)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\tilde z_*/\kappa^2$")
    ax.legend(h, l, loc="lower left", ncol=2, columnspacing=1.0)
    panel_label(ax, "(b)")
    save(fig, "fig4_kaniadakis")
    plt.close(fig)


# =============================================================================
# Fig. 3
# =============================================================================
ASY = dict(color="k", ls=(0, (3, 2)), lw=0.7, zorder=6)


def plot_fig3(plt, d, nums, kappas_shown):
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.5 * CM))
    ax = axs[0]
    al = d["a_alpha"]
    ax.plot(al, np.abs(d["a_R_bg"]), color="k", label="BG")
    for c, q in zip(COL, F3_QS):
        ax.plot(al, np.abs(d["a_R_%g" % q]), color=c, label=r"$q=%g$" % q)
        m = al >= 1.0
        gp = al[m] - F3_ZETA + 1.0 / (q - 1.0)          # gamma' = gamma + 1/(q-1), gamma = alpha - zeta
        ax.plot(al[m], C_tsallis(q) * gp ** -4, **ASY)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], **ASY)); l.append(r"$\mathcal{C}(q)\,\bar\gamma^{-4}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e5)
    ax.set_ylim(1e-21, 1)
    ax.set_xlabel(r"$\alpha=(m-\mu)/T$")
    ax.set_ylabel(r"$|\hat R_{\rm FD}|$")
    ax.legend(h, l, loc="lower left")
    panel_label(ax, "(a)")
    ax = axs[1]
    al = d["b_alpha"]
    for c, k in zip(COL, kappas_shown):
        lab = r"$\kappa=1/6$" if abs(k - 1 / 6) < 1e-9 else r"$\kappa=%.2f$" % k
        ax.plot(al, np.abs(d["b_R_%.6f" % k]), color=c, label=lab)
        g = al - F3_ZETA                                 # gamma = alpha - zeta
        m = g >= 3.0
        ax.plot(al[m], K_kaniadakis(k) * g[m] ** (1.0 / k - 6.0), **ASY)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], **ASY)); l.append(r"$\mathcal{K}(\kappa)\,\gamma^{1/\kappa-6}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_ylim(1e-13, 1e5)
    ax.set_xlabel(r"$\alpha=(m-\mu)/T$")
    ax.set_ylabel(r"$|\hat R|$ (classical)")
    ax.legend(h, l, loc="lower left")
    panel_label(ax, "(b)")
    save(fig, "fig3_limits")
    plt.close(fig)


# =============================================================================
# numbers + checks
# =============================================================================
def build_numbers(d1, d2, d3):
    nums = {"conventions": "Rhat = (g T^3/2pi^2) R, geometer sign (FD<0, BE>0); beta = 1, m = zeta; "
                           "alpha = -ln ztilde = (m-mu)/T; gamma = -mu/T = alpha - zeta; gamma' = gamma + 1/(q-1)"}
    zl = sorted(set((0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 30.0, 100.0)))
    nums["r0"] = {"%g" % z: r0(z) for z in zl}
    nums["r0_direct_limit_check"] = {"%g" % z: dict(closed=r0(z), direct_alpha60=float(v), rel=float(v / r0(z) - 1))
                                     for z, v in zip(d1["b_zchk"], d1["b_direct"])}
    nums["Gamma_B"] = {"%g" % z: Gamma_B(z) for z in (0.01, 0.1, 1.0, 10.0, 30.0)}
    nums["Gamma_B_over_r0"] = {"%g" % z: Gamma_B(z) / r0(z) for z in (0.1, 1.0, 10.0, 30.0)}
    nums["Upsilon"] = {"%g" % z: Upsilon(z) for z in (0.0, 0.1, 1.0, 10.0)}
    nums["A_kin"] = {"%g" % z: A_kin(z) for z in (0.0, 0.1, 1.0, 10.0)}
    nums["zstar_over_kappa2_limit"] = {"%g" % z: zstar_lead_coef(z) for z in (0.0, 0.1, 1.0, 10.0)}
    # Fig. 1(a) endpoints
    fa = {}
    for z in F1_ZETAS_A:
        for a in (1, -1):
            R = d1["a_R_%g_%d" % (z, a)]
            fa["zeta=%g,%s" % (z, "FD" if a == 1 else "BE")] = dict(
                Rhat_over_r0_at_ztilde_1em6=float(R[0] / r0(z)),
                Rhat_over_r0_at_alpha_min=float(R[-1] / r0(z)), alpha_min=float(d1["a_alpha"][-1]))
        fa["zeta=%g,FD at ztilde=1 (alpha=0)" % z] = float(tg.curvature_np(z, 0.0, tg.Stat("bg", 1)) / r0(z))
    nums["fig1a"] = fa
    # Fig. 1(c): where sqrt(alpha) R / Gamma_B is within 1 % / 10 % of 1
    fc = {}
    for z in F1_ZETAS_C:
        al = d1["c_alpha_%g" % z]
        y = np.sqrt(al) * d1["c_R_%g" % z] / float(d1["c_Gamma_%g" % z])
        out = {}
        for tol in (0.01, 0.1):
            bad = np.where(np.abs(y - 1) > tol)[0]
            out["alpha_max_within_%g" % tol] = float(al[bad[0] - 1]) if len(bad) and bad[0] > 0 else None
        out["value_at_alpha_over_zeta_1e-6"] = float(y[0])
        fc["zeta=%g" % z] = out
    nums["fig1c"] = fc
    # Fig. 2: crossovers
    ztab = {}
    for i, z in enumerate(F2B_ZETAS):
        row = {}
        for j, k in enumerate(d2["b_kappa"]):
            if any(abs(k - t) < 1e-9 for t in F2B_TABLE_KAPPAS):
                row["%.2f" % k] = float(math.exp(-d2["b_alpha_star"][i, j]))
        ztab["zeta=%g" % z] = row
    nums["kaniadakis_crossover"] = ztab
    nums["kaniadakis_crossover_nsignchanges_all_one"] = bool(np.all(d2["b_nsignchanges"] == 1))
    nums["zstar_over_kappa2_curve"] = {"kappa": [float(x) for x in d2["b_kappa"]],
                                       **{"zeta=%g" % z: [float(math.exp(-a) / k ** 2) for a, k in
                                                          zip(d2["b_alpha_star"][i], d2["b_kappa"])]
                                          for i, z in enumerate(F2B_ZETAS)}}
    nums["zstar_small_kappa_leading_zeta1"] = {"%.2f" % k: float(tg.zstar_small_kappa(k, 1.0)) for k in F2_KAPPAS}
    # Fig. 3 constants and deep values
    nums["C_q"] = {"%g" % q: C_tsallis(q) for q in F3_QS}
    nums["C_q_over_p4"] = {"%g" % q: C_tsallis(q) * (q - 1) ** 4 for q in F3_QS}
    nums["K_kappa"] = {("1/6" if abs(k - 1 / 6) < 1e-9 else "%.2f" % k): K_kaniadakis(k) for k in F3_KAPPAS}
    rat = {}
    al = d3["a_alpha"]
    for q in F3_QS:
        gp = al - F3_ZETA + 1 / (q - 1)
        rr = -d3["a_R_%g" % q] * gp ** 4 / C_tsallis(q)
        rat["q=%g" % q] = dict(alpha=[float(al[i]) for i in (0, 60, 100, -1)],
                               ratio_to_asymptote=[float(rr[i]) for i in (0, 60, 100, -1)],
                               Rhat_at_alpha_1e5=float(d3["a_R_%g" % q][-1]))
    rat["BG FD at alpha=1e5 / r0(1)"] = float(-d3["a_R_bg"][-1] / r0(1.0))
    nums["fig3a"] = rat
    rat = {}
    al = d3["b_alpha"]
    for k in F3_KAPPAS:
        g = al - F3_ZETA
        R = d3["b_R_%.6f" % k]
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = -R / (K_kaniadakis(k) * g ** (1 / k - 6))
        rat["kappa=%.4f" % k] = dict(exponent=1 / k - 6, alpha=[float(al[i]) for i in (20, 60, -1)],
                                     ratio_to_asymptote=[float(rr[i]) for i in (20, 60, -1)],
                                     Rhat_at_alpha_1=float(R[0]), Rhat_at_alpha_1e5=float(R[-1]),
                                     Rhat_range=[float(np.abs(R).min()), float(np.abs(R).max())])
    nums["fig3b"] = rat
    return nums


def checks(nums):
    out = {}

    def chk(name, val, ref, tol):
        rel = abs(val / ref - 1)
        out[name] = dict(value=val, report=ref, rel=rel, ok=bool(rel < tol))
        print("  check %-20s %.10g  report %.10g  rel %.1e  %s" % (name, val, ref, rel, "ok" if rel < tol else "MISMATCH"))

    chk("r0(1)", nums["r0"]["1"], REPORT["r0(1)"], 1e-12)
    chk("r0(0)", nums["r0"]["0"], REPORT["r0(0)"], 1e-12)
    for z in ("0.1", "1", "10", "30"):
        chk("Gamma_B(%s)" % z, nums["Gamma_B"][z], REPORT["Gamma_B(%s)" % z], 1e-9)
    for z in ("0.1", "1", "10"):
        for k in ("0.05", "0.10", "0.15", "0.20", "0.24"):
            key = "zstar(%s,%s)" % ({"0.10": "0.1", "0.20": "0.2"}.get(k, k), z)
            chk(key, nums["kaniadakis_crossover"]["zeta=%s" % z][k], REPORT[key], 5e-5)
        chk("lim zstar/k2 (%s)" % z, nums["zstar_over_kappa2_limit"][z], REPORT["lim zstar/k2 (%s)" % z], 1e-5)
    chk("C(1.1)", C_tsallis(1.1), REPORT["C(1.1)"], 1e-9)
    chk("C(1.2)", C_tsallis(1.2), REPORT["C(1.2)"], 1e-12)
    for k, key in ((0.24, "K(0.24)"), (0.2, "K(0.2)"), (1 / 6, "K(1/6)"), (0.1, "K(0.1)")):
        chk(key, K_kaniadakis(k), REPORT[key], 1e-9)
    return out


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="ignore the curve caches")
    ap.add_argument("--kappa3b", type=float, default=0.14,
                    help="growing-branch kappa (< 1/6) shown in Fig. 3(b); cached: 0.10 or 0.14")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    t0 = time.time()
    print("Fig. 1 data")
    d1 = load_or("fig1", compute_fig1, args.recompute)
    print("Fig. 2 data")
    d2 = load_or("fig2", compute_fig2, args.recompute)
    print("Fig. 3 data")
    d3 = load_or("fig3", compute_fig3, args.recompute)
    print("numbers")
    nums = build_numbers(d1, d2, d3)
    nums["checks_against_report"] = checks(nums)
    kshown = (args.kappa3b, 1.0 / 6.0, 0.20, 0.24)
    nums["fig3b_kappas_shown"] = list(kshown)
    with open(os.path.join(DATA, "figures_numbers.json"), "w") as fh:
        json.dump(nums, fh, indent=1)
    plt = setup_mpl()
    plot_fig1(plt, d1, nums)
    plot_fig2(plt, d2, nums)
    plot_fig3(plt, d3, nums, kshown)
    print("done in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
