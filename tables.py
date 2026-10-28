# -*- coding: utf-8 -*-
r"""
tables.py -- closed-form tables for the paper (D = 3, reduced units, T = 1)
===========================================================================

  c1(zeta), Xi(zeta), b2(zeta) = B_2^red(a=+1), Y(zeta), Y/(2 Xi), C_BE(zeta),
  Tsallis deep-dilute constant Cq(q), Kaniadakis deep-dilute constant Kk(kappa),
  small-kappa crossover coefficient  ztilde*/kappa^2 -> Y/(c1 A_kin).
Physical units:  R = (2 pi^2/g) T^{-3} Rhat,  B_2 = (2 pi^2/g) T^{-3} b2 * a.
Output: data/tables.json (and printed).
"""
import json
import os

import mpmath as mp

import thermogeom as tg
from verify_deformed import Cq_tsallis, Kk_kaniadakis

HERE = os.path.dirname(os.path.abspath(__file__))


def C_BE(zeta, dps=40):
    """Coefficient of the Bose divergence Rhat ~ C_BE alpha^{-1/2} (alpha = -ln ztilde -> 0)."""
    with mp.workdps(dps + 10):
        mo = tg.moments_mp(1, 0, zeta, tg.Stat("bg", -1), red="none", dps=dps)
        P, Q = mo["row1"][0], mo["row1"][1]
        A3, B3 = mo["row2"][0], mo["row2"][1]
        return (P * B3 - Q * A3) / P ** 2 * mp.power(2 * mp.mpf(zeta), -1.5) / mp.pi


def main():
    out = {"zeta_table": [], "tsallis": [], "kaniadakis": []}
    print("%7s %14s %12s %12s %12s %10s %12s %14s %12s" %
          ("zeta", "c1", "Xi", "b2", "Y", "Y/(2Xi)", "A_kin", "Y/(c1 A_kin)", "C_BE"))
    with mp.workdps(40):
        for z in (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 16.0, 32.0, 100.0):
            c1 = tg.c1_closed(z)
            Xi = tg.Xi_dilute(z)
            b2 = tg.B2_reduced(z, 1)
            Y = tg.Y_kappa(z)
            A = tg.A_reduced(z, kinetic=True)
            cbe = C_BE(z) if z > 0 else mp.inf
            row = dict(zeta=z, c1=float(c1), Xi=float(Xi), b2=float(b2), Y=float(Y), Y_over_2Xi=float(Y / (2 * Xi)),
                       A_kin=float(A), zstar_coef=float(Y / (c1 * A)), C_BE=float(cbe) if z > 0 else None,
                       C_BE_zeta32=float(cbe * mp.power(z, 1.5)) if z > 0 else None)
            out["zeta_table"].append(row)
            print("%7g %14.10f %12.9f %12.9f %12.9f %10.6f %12.6f %14.6f %12.6g" %
                  (z, c1, Xi, b2, Y, Y / (2 * Xi), A, Y / (c1 * A), cbe if z > 0 else float("inf")))
        lim = 90 * mp.zeta(3) / (mp.power(2, 3.5) * mp.pi ** 5)
        print("UR limit of C_BE zeta^{3/2} = 90 zeta(3)/(2^{7/2} pi^5) = %s ; NR limit 2^{-1/2} zeta(3/2)/(5 pi zeta(5/2)) = %s"
              % (mp.nstr(lim, 12), mp.nstr(mp.zeta(1.5) / (5 * mp.pi * mp.zeta(2.5) * mp.sqrt(2)), 12)))
        out["C_BE_zeta32_limits"] = dict(UR=float(lim), NR=float(mp.zeta(1.5) / (5 * mp.pi * mp.zeta(2.5) * mp.sqrt(2))))
    print("\nTsallis deep-dilute constant Cq (Rhat -> -a Cq beta^3 gamma'^-4);  Cq/p^4 -> 1/16 as q->1")
    for q in (1.001, 1.01, 1.02, 1.05, 1.1, 1.15, 1.2, 1.24):
        p = 1 / (q - 1)
        c = Cq_tsallis(q)
        print("   q=%-6g p=%-8.3f Cq=%.8g   Cq/p^4=%.6f" % (q, p, c, c / p ** 4))
        out["tsallis"].append(dict(q=q, Cq=c, Cq_over_p4=c / p ** 4))
    print("\nKaniadakis deep-dilute constant Kk (Rhat -> -Kk beta^3 gamma^{1/kappa-6})")
    for k in (0.01, 0.02, 0.05, 0.1, 0.15, 1.0 / 6.0, 0.2, 0.24):
        K = Kk_kaniadakis(k)
        print("   kappa=%-8.5f exponent 1/kappa-6 = %-8.3f  Kk=%s" % (k, 1 / k - 6, mp.nstr(K, 10)))
        out["kaniadakis"].append(dict(kappa=k, exponent=1 / k - 6, Kk=float(K)))
    with open(os.path.join(HERE, "data", "tables.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
