"""Check: nucleon-antinucleon gas (MJ) at mu_B=0 has R<0; adding baryon-neutral pions can flip the sign.
phi(beta,gamma) = 2 cosh(gamma) A_N(beta) + phi_pi(beta); curvature from Eq. (R) by mpmath differentiation."""
import mpmath as mp

mp.mp.dps = 40
mN, gN = mp.mpf("0.938"), 4
mpi, gpi = mp.mpf("0.138"), 3


def A_N(beta):  # MJ one-particle function per volume (GeV^3)
    return gN * mN**2 * mp.besselk(2, beta * mN) / (2 * mp.pi**2 * beta)


def phi_pi(beta):  # BE pions, mu=0
    f = lambda p: p**2 * (-mp.log(1 - mp.exp(-beta * mp.sqrt(p**2 + mpi**2))))
    return gpi / (2 * mp.pi**2) * mp.quad(f, [0, 1, 5, mp.inf])


def curvature(phi, beta, gamma):
    d = lambda i, j: mp.diff(phi, (beta, gamma), (i, j))
    M = mp.matrix([[d(2, 0), d(1, 1), d(0, 2)],
                   [d(3, 0), d(2, 1), d(1, 2)],
                   [d(2, 1), d(1, 2), d(0, 3)]])
    detg = d(2, 0) * d(0, 2) - d(1, 1) ** 2
    return -mp.det(M) / (2 * detg**2)


for T in [mp.mpf("0.100"), mp.mpf("0.150")]:
    b = 1 / T
    nuc = lambda be, ga: 2 * mp.cosh(ga) * A_N(be)
    mix = lambda be, ga: 2 * mp.cosh(ga) * A_N(be) + phi_pi(be)
    print("T=%s GeV  R(nucleons)=%s GeV^-3  R(nucleons+pions)=%s GeV^-3" %
          (T, mp.nstr(curvature(nuc, b, 0), 6), mp.nstr(curvature(mix, b, 0), 6)))
