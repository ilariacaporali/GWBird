import numpy as np
from gwbird import detectors
from gwbird.overlap import Response
from gwbird.utils import H0

def SNR(Tobs, f, gw_params, detectors_list, pol, psi=0, shift_angle=None, gw_spectrum_func=None):
    """
    Calculate the signal-to-noise ratio for a given GW signal and multiple detector pairs,
    allowing either an array of GW parameters or a custom function for the spectrum.
    
    Parameters
    - Tobs : float (Observation time in years)
    - f : array_like (Frequency of the GW signal)
    - gw_params : array-like or None (if array-like, it should be [log10A_gw, n_gw, fref], if None you have to consider a gw_spectrum_func later)
    - detectors_list : list of str (list of detector names)
    - pol : str (polarization of the GW signal)
    - psi : float, optional (polarization angle of the GW signal (default is 0))
    - shift_angle : float, optional (shift angle used if one of the detectors is ET_L2)
    - gw_spectrum_func : function, optional (Custom function defining the GW energy spectrum Omega_GW(f))

    Returns
    - float (computed signal-to-noise ratio (SNR))
    """
    
    def Omega_GW(f):
        if callable(gw_spectrum_func):
            return gw_spectrum_func(f)
        logA_gw, n_gw, fref = gw_params
        return 10**logA_gw * (f / fref) ** n_gw
    
    Tobs = Tobs * 365 * 24 * 3600  # Convert years to seconds
    
    if len(detectors_list) == 1 and detectors_list[0].upper() == "LISA":
        Omega_gw = Omega_GW(f)
        total_integral = 0
        
        channels = ['A', 'E', 'T']
        for channel in channels:
            overlap = Response.overlap(f'LISA {channel}', f'LISA {channel}', f, psi, pol)
            noise = detectors.LISA_noise_AET(f, channel)
            integrand = (Omega_gw * overlap) ** 2 / (f ** 6 * noise**2)
            total_integral += np.trapezoid(integrand, f)
        
        snr = 3 * H0**2 / (10 * np.pi**2) * np.sqrt(2 * total_integral * Tobs)
        return snr
    
    if len(detectors_list) == 1 and detectors_list[0].lower() == "pulsars":
        
        def PTA_Pn():
            DT = (365*24*3600)/20  # s
            s = 100 * 1e-9  # s
            return 2 * (s**2) * DT

        def PTA_Sn(f):
            f = np.asarray(f)
            mask = f >= 8e-9
            return np.where(mask, PTA_Pn() * 12 * (np.pi**2) * f**2, 1)
        
        N, p, D = detectors.get_NANOGrav_pulsars()
        integrand = 0
        for i in range(N):
            for j in range(i + 1, N): 
                overlap = Response.pairwise_overlap(f, p[i], p[j],D[i], D[j], pol, psi)
                integrand += (overlap * Omega_GW(f)) ** 2 / (f ** 6 * PTA_Sn(f)**2)
        total_integral = np.trapezoid(integrand, f)
        snr = 3 * H0**2 / (2 * np.pi**2) * np.sqrt(2 * total_integral * Tobs)
        return snr
    
    total_integral = 0
    
    for i in range(len(detectors_list)):
        for j in range(i + 1, len(detectors_list)):
            det1 = detectors_list[i]
            det2 = detectors_list[j]
            
            if "ET L2" in [det1, det2]:
                orf = Response.overlap(det1, det2, f, psi, pol, shift_angle)
            else:
                orf = Response.overlap(det1, det2, f, psi, pol)
            
            fI, PnI = detectors.detector_Pn(det1)
            fII, PnII = detectors.detector_Pn(det2)
            
            Ni = np.interp(f, fI, PnI)
            Nj = np.interp(f, fII, PnII)
            
            Omega_gw = Omega_GW(f)
            integrand = (orf * Omega_gw) ** 2 / (f ** 6) / (Ni * Nj)
            integral = np.trapezoid(integrand, f)
            total_integral += integral
    
    snr = 3 * H0**2 / (10 * np.pi**2) * np.sqrt(2 * total_integral * Tobs)
    return snr
