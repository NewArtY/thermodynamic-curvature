# -*- coding: utf-8 -*-
r"""
compare_old_code.py -- compare the OLD Annals code (code/figures/fig06_ricci.py, read-only)
with the new independent implementation (thermogeom.py).

Old conventions (established by reading the old code): T = 1, m = zeta, measure p^2 dp
(g/(2 pi^2) dropped), argument z of ricci_scalar / ricci_deformed is ztilde = e^{(mu-m)/T};
momentum cut-off p_max = sqrt((m+90)^2 - m^2) for ALL statistics (also power-law tails).
The old code is the archive 10.5281/zenodo.21383735; point the environment variable OLD_CODE
to its unpacked root (default: ../../code).
Outputs data/old_code_comparison.json
"""
import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np

import thermogeom as tg

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.abspath(os.environ.get("OLD_CODE", os.path.join(HERE, "..", "..", "code")))
sys.path.insert(0, OLD)
sys.path.insert(0, os.path.join(OLD, "figures"))
import fig06_ricci as old  # noqa: E402  (module-level code only defines functions)


def main():
    res = {}
    print("1) BG curvature (old ricci_scalar vs new), zeta = 1")
    rows = []
    for a in (1, -1):
        for zt in (1e-3, 1e-2, 0.1, 0.5, 0.9, 0.999):
            Ro = old.ricci_scalar(zt, a, m=1.0)
            with mp.workdps(40):
                Rn = float(tg.curvature_mp(1.0, -mp.log(zt), tg.Stat("bg", a), dps=30))
            rows.append(dict(a=a, ztilde=zt, old=Ro, new=Rn, rel=abs(Ro / Rn - 1)))
            print("   a=%+d ztilde=%-6g old=%.10e new=%.10e rel=%.1e" % (a, zt, Ro, Rn, abs(Ro / Rn - 1)))
    res["bg"] = rows

    print("2) c1 (old c1_of_zeta vs new closed form)")
    rows = []
    for z in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
        co = old.c1_of_zeta(z)
        cn = float(tg.c1_closed(z))
        rows.append(dict(zeta=z, old=co, new=cn, rel=abs(co / cn - 1)))
        print("   zeta=%-4g old=%.10f new=%.10f rel=%.1e" % (z, co, cn, abs(co / cn - 1)))
    res["c1"] = rows

    print("3) deformed curvature (old ricci_deformed vs new), zeta = 1")
    rows = []
    for sigma, par in (("q", 1.1), ("q", 1.2), ("q", 1.24), ("k", 0.05), ("k", 0.15), ("k", 0.2), ("k", 0.24)):
        for a in (1, 0, -1):
            for zt in (1e-3, 0.1, 0.9):
                Ro = old.ricci_deformed(zt, a, sigma, par, m=1.0)
                with mp.workdps(40):
                    Rn = float(tg.curvature_mp(1.0, -mp.log(zt), tg.Stat(sigma, a, par), dps=30))
                rel = abs(Ro - Rn) / max(abs(Rn), 1e-300) if Rn != 0 else abs(Ro)
                rows.append(dict(sigma=sigma, par=par, a=a, ztilde=zt, old=Ro, new=Rn, rel=rel))
                print("   %s=%-5g a=%+d ztilde=%-6g old=% .8e new=% .8e  %s" %
                      (sigma, par, a, zt, Ro, Rn, ("rel %.1e" % rel) if Rn != 0 else "(new exactly 0)"))
    res["deformed"] = rows

    print("4) old crossover z*(kappa) (old build_deformed_curves, zeta=1) vs new mp roots")
    t0 = time.time()
    kap = [0.05, 0.10, 0.15, 0.20]
    _, zst = old.build_deformed_curves(kap, 1.0)
    rows = []
    for k in kap:
        st = tg.Stat("k", -1, k)
        al = np.linspace(0.05, 12, 400)
        R = [tg.curvature_np(1.0, x, st) for x in al]
        sc = tg.find_sign_changes(al, R)
        roots = []
        for i in sc:
            with mp.workdps(40):
                r = mp.findroot(lambda x: tg.curvature_mp(1.0, x, st, dps=30), (mp.mpf(al[i]), mp.mpf(al[i + 1])),
                                solver="illinois", tol=mp.mpf(10) ** -24)
            roots.append(math.exp(-float(r)))
        rows.append(dict(kappa=k, old_zstar=zst[k], new_ztilde_star=roots))
        print("   kappa=%.2f old z*=%s new ztilde*=%s" % (k, zst[k], ["%.6f" % v for v in roots]))
    print("   (%.0fs)" % (time.time() - t0))
    res["zstar"] = rows
    with open(os.path.join(HERE, "data", "old_code_comparison.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
