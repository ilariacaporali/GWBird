import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from nest import overlap_try as overlap
from nest import detectors   
from astropy.cosmology import Planck18
from nest import pls_try as pls
import matplotlib.cm as cm
from scipy.integrate import simps

cosmo = Planck18
H0 =  cosmo.H0.to('1/s').value
h = 0.7


def Seff(f, fref, snr, Tobs, orf, Ni, Nj):
    return np.sqrt(Ni*Nj)/orf

def Omega_eff(f, fref, snr, Tobs, orf, Ni, Nj):
    return 10* np.pi**2 /(3* (H0**2)/(h**2))* f**3 * Seff(f, fref, snr, Tobs, orf, Ni, Nj)
    
def Omega_beta(f, fref, snr, Tobs, beta, orf, Ni, Nj):
    Tobs = Tobs * 365 * 24 * 3600
    Omega_eff_num = Omega_eff(f, fref, snr, Tobs, orf, Ni, Nj)
    integrand = (((f/fref)**(beta)) / (Omega_eff_num))**2
    integral = simps(integrand, f)
    return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)

def Omega_GW(f, fref, snr, Tobs, beta, orf, Ni, Nj):
    return Omega_beta(f, fref, snr, Tobs, beta, orf, Ni, Nj) * ((f/fref)**(beta))

def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, orf, Ni, Nj):
    beta = np.linspace(beta_min, beta_max, 1000)
    Omega = []
    for i in range(len(beta)):
        Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], orf, Ni, Nj))     
    return beta, np.array(Omega)

def PLS(which_det1, which_det2, f, fref, pol, snr, Tobs, beta_min, beta_max, shift_angle):

    fi, PnI = detectors.detector_Pn(which_det1)
    fj, PnJ = detectors.detector_Pn(which_det2)

    PnI = np.interp(f, fi, PnI)
    PnJ = np.interp(f, fj, PnJ)

    if (which_det1 == 'LISA 1' and which_det2 == 'LISA 1') or (which_det1 == 'LISA 2' and which_det2 == 'LISA 2') or (which_det1 == 'LISA 3' and which_det2 == 'LISA 3'):
        XX = overlap.overlap('LISA 1', 'LISA 1', f, 0, pol)#[0]  # auto
        print(XX[0])
        XY = overlap.overlap('LISA 1', 'LISA 2', f, 0, pol)#[0]  # cross
        print(XY[0])
        # the overlap is evaluated in the diagonal basis
        orfIJ = (np.array(XX) - np.array(XY))*(5/2)
        print(orfIJ[0])
    
    else:
        orfIJ = overlap.overlap(which_det1, which_det2, f, 0 , pol, shift_angle )

    beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, orfIJ, PnI, PnJ)
    print(Omega.shape)
    pls = np.zeros(len(f))
    for i in range(len(f)):
        pls[i] = np.max(Omega[:,i])
    return pls
