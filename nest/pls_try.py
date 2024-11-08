import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from nest import overlap_try as overlap
from nest import detectors   
from astropy.cosmology import Planck15

cosmo = Planck15
H0 =  cosmo.H0.to('1/s').value

h = 0.7

def Omega_eff(f, PnI, PnJ, orfIJ):
    return 10* np.pi**2 /(3* (H0**2)/(h**2)) * np.sqrt((f**6) *PnI * PnJ / (orfIJ**2))

def Omega_beta(f_range, PnI, PnJ, orfIJ, beta, fref, snr, Tobs):
    Tobs = Tobs*364*24*3600
    integrand = lambda f : ((f/fref)**(2*beta))/ (Omega_eff(f, PnI, PnJ, orfIJ)**2)
    integral = np.trapz(integrand(f_range), f_range)
    return snr / np.sqrt(2*Tobs*integral)

def Omega_GW(f_i, PnI, PnJ, orfIJ, beta, fref, snr, Tobs):
    return Omega_beta(f_i, PnI, PnJ, orfIJ, beta, fref, snr, Tobs) * ((f_i/fref)**(beta))

def all_Omega_GW(f_i, PnI, PnJ, orfIJ, beta_min, beta_max, fref, snr, Tobs):
    beta = np.linspace(beta_min, beta_max, 1000)
    Omega = []
    for i in range(len(beta)):
        Omega.append(Omega_GW(f_i, PnI, PnJ, orfIJ, beta[i], fref, snr, Tobs))
        
    return beta, np.array(Omega)

def find_pls(which_det1, which_det2, f, pol,  beta_min, beta_max, fref, snr, Tobs, shift_angle):
    
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
        orfIJ = (np.array(XX) - np.array(XY))
        print(orfIJ[0])

    else: 
        orfIJ = overlap.overlap(which_det1, which_det2, f, 0 , pol, shift_angle)
    
    beta, Omega = all_Omega_GW(f, PnI, PnJ, orfIJ, beta_min, beta_max, fref, snr, Tobs)
    
    pls = np.zeros(len(f))

    for i in range(len(f)):
        pls[i] = np.max(Omega[:,i])
    return pls


def find_pls_weighted(which_det1, which_det2, f, pol, beta_min, beta_max, fref, snr, Tobs, shift_angle):
    pls1 = find_pls(which_det1, which_det2, f, pol, beta_min, beta_max, fref, snr, Tobs, shift_angle)
    return 1/( 3/pls1)



#************************* PTA *************************#

def PTA_Pn():
    DT_inverse = 20/(365*24*3600) #Hz
    s = 100 * 1e-9 #s
    return 2* (s**2)/DT_inverse


def PTA_Sn(f):
    f = np.asarray(f) # Ensure f is a NumPy array
    mask = f >= 8e-9 # Create a boolean mask where True indicates elements greater than or equal to 8e-9
    return np.where(mask, PTA_Pn() * 12 * (np.pi**2) * f**2, 1) # Apply the mask to the result

def PTA_Seff(f, catalogue):
    hd = 0
    for i in range(len(catalogue)):
        for j in range(i+1, len(catalogue)):
            hd += (overlap_old.HellingsDowns(catalogue[i], catalogue[j]))**2
    return hd*PTA_Sn(f)

def PTA_Omegaeff(f, catalogue, om_load):
    if om_load==False:
        return 10 * np.pi * np.pi * f**3 * PTA_Seff(f, catalogue) / (3* (H0**2))
    else:
        return np.genfromtxt('/home/ilaria/Desktop/morfeus/pypack/Omegaeff.txt', unpack=True, usecols=1)

def Omega_beta_PTA(f_range, snr, Tobs, beta, catalogue, om_load):
    Tobs = Tobs*365*24*3600
    fref = 1e-8
    integrand = lambda f : ((f/fref)**(2*beta))/ (PTA_Omegaeff(f, catalogue, om_load)**2)
    integral = np.trapz(integrand(f_range), f_range)
    return snr / np.sqrt(2*Tobs*integral)

def Omega_GW_PTA(f_i,  beta, fref, snr, Tobs, catalogue, om_load):
    return Omega_beta_PTA(f_i, snr, Tobs, beta, catalogue, om_load) * ((f_i/fref)**(beta))

def all_Omega_GW_PTA(f_i, snr, Tobs, beta_min, beta_max, catalogue, om_load):
    beta = np.linspace(beta_min, beta_max, 100)
    fref = 1e-8
    Omega = np.zeros((len(beta), len(f_i)))
    for i, beta_val in enumerate(beta):
        Omega[i, :] = Omega_GW_PTA(f_i, beta_val, fref, snr, Tobs, catalogue, om_load)
    return beta, Omega

def find_pls_PTA(f_i, snr, Tobs, beta_min, beta_max, catalogue, om_load):

    beta, Omega = all_Omega_GW_PTA(f_i, snr, Tobs, beta_min, beta_max, catalogue, om_load)
    pls = np.max(Omega, axis=0)
    return pls

        
    

 