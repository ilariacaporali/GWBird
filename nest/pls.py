import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from nest import overlap as overlap
from nest import detectors   
from nest import nell
from nest.detectors import LISA_noise_AET
from nest.overlap import Response
import matplotlib.cm as cm
from scipy.integrate import simps
from nest.utils import c, H0, h



def S_eff(f, fref, snr, Tobs, orf, Ni, Nj):

    '''
    Effective noise power spectral density
    '''

    return np.sqrt(Ni*Nj)/np.abs(orf)

def Omega_eff(f, fref, snr, Tobs, orf, Ni, Nj):
    
    '''
    Effective energy density
    '''

    return 10* np.pi**2 /(3* (H0**2)/(h**2))* f**3 * S_eff(f, fref, snr, Tobs, orf, Ni, Nj)
    
def Omega_beta(f, fref, snr, Tobs, beta, orf, Ni, Nj):

    '''
    Energy density for a given beta
    '''

    Tobs = Tobs * 365 * 24 * 3600
    Omega_eff_num = Omega_eff(f, fref, snr, Tobs, orf, Ni, Nj)
    integrand = (((f/fref)**(beta)) / (Omega_eff_num))**2
    integral = simps(integrand, f)
    return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)

def Omega_GW(f, fref, snr, Tobs, beta, orf, Ni, Nj):

    '''
    Power spectral density of the GW signal
    '''

    return Omega_beta(f, fref, snr, Tobs, beta, orf, Ni, Nj) * ((f/fref)**(beta))

def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, orf, Ni, Nj):

    '''
    Power spectral density of the GW signal for a range of beta values
    '''

    beta = np.linspace(beta_min, beta_max, 1000)
    Omega = []
    for i in range(len(beta)):
        Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], orf, Ni, Nj))     
    return beta, np.array(Omega)

def PLS(det1, det2, f, fref, pol, snr, Tobs, beta_min, beta_max, shift_angle):

    '''
    det1, det2: detectors (string)
    f: frequency array (array float)
    fref: reference frequency (float)
    pol: polarization mode (string)
    snr: signal-to-noise ratio threshold (float)
    Tobs: observation time (float) - in YEARS
    beta_min, beta_max: range of beta values (float)
    shift_angle: shift angle (None or float)
    '''

    fi, PnI = detectors.detector_Pn(det1)
    fj, PnJ = detectors.detector_Pn(det2)

    PnI = np.interp(f, fi, PnI)
    PnJ = np.interp(f, fj, PnJ)

    if (det1 == 'LISA 1' and det2 == 'LISA 1') or (det1 == 'LISA 2' and det2 == 'LISA 2') or (det1 == 'LISA 3' and det2 == 'LISA 3'):
        XX = overlap.Response.overlap('LISA 1', 'LISA 1', f, 0, pol) # auto
        XY = overlap.Response.overlap('LISA 1', 'LISA 2', f, 0, pol) # cross
        # the overlap is evaluated in the diagonal basis
        orfIJ = (np.array(XX) - np.array(XY))*(5/2)

    else:
        orfIJ = overlap.Response.overlap(det1, det2, f, 0 , pol, shift_angle )

    beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, orfIJ, PnI, PnJ)

    pls = np.zeros(len(f))
    for i in range(len(f)):
        pls[i] = np.max(Omega[:,i])
    return pls

def PLS_LISA(f, fref, pol, snr, Tobs, beta_min, beta_max, shift_angle):

    psd_A = LISA_noise_AET(f, 'A')
    psd_E = LISA_noise_AET(f, 'E')
    psd_T = LISA_noise_AET(f, 'T')

    R_AA = R_EE = Response.overlap_AET('AA', f, 0, pol)
    R_TT = Response.overlap_AET('TT', f, 0, pol)

    _, Omega_A = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, R_AA, psd_A, psd_A)
    _, Omega_E = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, R_EE, psd_E, psd_E)
    _, Omega_T = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, R_TT, psd_T, psd_T)

    pls_A = np.zeros(len(f))
    pls_E = np.zeros(len(f))
    pls_T = np.zeros(len(f))

    for i in range(len(f)):
        pls_A[i] = np.max(Omega_A[:,i])
        pls_E[i] = np.max(Omega_E[:,i])
        pls_T[i] = np.max(Omega_T[:,i])

    return 1/np.sqrt(1/ pls_A**2 + 1/ pls_E**2 + 1/ pls_T**2)