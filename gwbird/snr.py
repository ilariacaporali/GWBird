import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec

from gwbird import detectors
from gwbird.skymap import Basis, Skymaps
from gwbird import pls
from gwbird.overlap import Response
from gwbird.nell import AngularResponse, Sensitivity_ell
from gwbird import nell
from gwbird.utils import c, H0, h

def SNR(Tobs, f, logA_gw, n_gw, fref, detectors_list, pol, psi=0, shift_angle=None):
    """
    Calculate the signal-to-noise ratio for a given GW signal and multiple detector pairs.
    
    Parameters
    ----------
    Tobs : float
        Observation time in years.
    f : float
        Frequency of the GW signal.
    logA_gw : float
        Logarithm of the amplitude of the GW signal.
    n_gw : float
        GW signal frequency index.
    fref : float
        Reference frequency for the GW signal.
    detectors_list : list of str
        List of detector names.
    pol : str
        Polarization of the GW signal.
    psi : float, optional
        Polarization angle of the GW signal.
    shift_angle : float, optional
        Shift angle of the GW signal (used only if one of the detectors is ET_L2).
    
    Returns
    -------
    float
        Signal-to-noise ratio.
    """
    
    def Omega_GW(f, logA_gw, n_gw, fref):
        return 10**logA_gw * (f / fref) ** n_gw 
    
    Tobs = Tobs * 365 * 24 * 3600  # Convert years to seconds
    
    # Special case for LISA-only detection
    if len(detectors_list) == 1 and detectors_list[0].upper() == "LISA":
        Omega_gw = Omega_GW(f, logA_gw, n_gw, fref)
        total_integral = 0
        
        channels = ['A', 'E', 'T']
        for channel in channels:
            overlap = Response.overlap(f'LISA{channel}', f'LISA{channel}', f, psi, pol)
            noise = detectors.LISA_noise_AET(f, channel)
            integrand = (Omega_gw * overlap) ** 2 / (f ** 6 * noise**2)
            total_integral += np.trapezoid(integrand, f)
        
        snr = 3 * H0**2 / (10 * np.pi**2) * np.sqrt(2 * total_integral * Tobs)
        return snr
    
    if len(detectors_list) == 1 and detectors_list[0].lower() == "pulsars":
        
        def PTA_Pn():
            DT = (365*24*3600)/20 # s
            s = 100 * 1e-9 #s
            return 2* (s**2) * DT

        def PTA_Sn(f):
            f = np.asarray(f) # Ensure f is a NumPy array
            mask = f >= 8e-9 # Create a boolean mask where True indicates elements greater than or equal to 8e-9
            return np.where(mask, PTA_Pn() * 12 * (np.pi**2) * f**2, 1) # Apply the mask to the result
        
        N, p, _= detectors.get_NANOGrav_pulsars()
        integrand = 0
        for i in range(N):
            for j in range(i + 1, N):
                overlap = Response.pairwise_overlap(f, p[i], p[j], pol, psi)
                integrand += (overlap * Omega_GW(f, logA_gw, n_gw, fref)) ** 2 / (f ** 6 * PTA_Sn(f)**2)
        total_integral = np.trapezoid(integrand, f)
        snr = 3 * H0**2 / (2 * np.pi**2) * np.sqrt(2 * total_integral * Tobs)
        return snr

    total_integral = 0
    
    # Iterate over all pairs of detectors
    for i in range(len(detectors_list)):
        for j in range(i + 1, len(detectors_list)):
            det1 = detectors_list[i]
            det2 = detectors_list[j]
            
            # Use shift_angle only if one of the detectors is ET_L2
            if "ET L2" in [det1, det2]:
                orf = Response.overlap(det1, det2, f, psi, pol, shift_angle)
            else:
                orf = Response.overlap(det1, det2, f, psi, pol)
            
            # Get the detector noise
            fI, PnI = detectors.detector_Pn(det1)
            fII, PnII = detectors.detector_Pn(det2)
            
            # Interpolate the noise power spectral density
            Ni = np.interp(f, fI, PnI)
            Nj = np.interp(f, fII, PnII)
            
            # Calculate the contribution from this detector pair
            Omega_gw = Omega_GW(f, logA_gw, n_gw, fref)
            integrand = (orf * Omega_gw) ** 2 / (f ** 6) / (Ni * Nj)
            integral = np.trapezoid(integrand, f)
            total_integral += integral
    
    snr = 3 * H0**2 / (10 * np.pi**2) * np.sqrt(2 * total_integral * Tobs)
    
    return snr