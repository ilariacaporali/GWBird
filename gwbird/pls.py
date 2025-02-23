import numpy as np
from gwbird import overlap as overlap
from gwbird import detectors   
from gwbird.detectors import LISA_noise_AET
from gwbird.overlap import Response
from gwbird.utils import c, H0, h


def PLS(det1, det2, f, fref, pol, snr, Tobs, psi, Stokes_parameter=False, shift_angle=False, fI=None, PnI=None, fJ=None, PnJ=None):
    '''
    Evaluate the sensitivity of a pair of detectors or a network of detectors to a Gravitational Wave Background (GWB) signal.

    Parameters:
    - det1, det2 : str or list of str
        The names of the detectors or the network of detectors to be considered.
        Supported options are:
        - Predefined detector networks such as 'ET triangular', 'LISA', or custom detectors defined as lists with specific parameters.
        - For custom detectors, provide a list with the following elements: [c, xA, xB, l, name]
          - c: array_like of length 3 (position of the detector in the Earth-centered frame in meters)
          - xA: array_like of length 3 (unit vector pointing towards the detector in the Earth-centered frame)
          - xB: array_like of length 3 (unit vector pointing towards the detector in the Earth-centered frame)
          - l: float (length of the detector arm in meters)
          - name: str (name of the detector)

    - f : array_like
        Frequency values (in Hz) for the Gravitational Wave signal.
    - fref : float
        Reference frequency (in Hz) for the signal analysis.
    - pol : str
        Polarization of the signal. Should be one of:
        - 't' for tensor polarization
        - 'v' for vector polarization
        - 's' for scalar polarization
    - snr : float
        Signal-to-noise ratio (SNR) threshold for the analysis.
    - Tobs : float
        Total observation time in years.
    - psi : float
        Polarization angle of the Gravitational Wave signal (in radians).
    - shift_angle : bool or float, optional
        If set to `True` or a specific angle value, shifts the angle of the response (used in the case of ET 2L).
    - fI, PnI, fJ, PnJ : optional
        Frequency arrays and corresponding Power Spectral Densities (PSDs) for custom detectors. These are used when no predefined detector networks are chosen.

    Returns:
    - PLS : array
        The sensitivity of the detector(s) to the GWB signal, evaluated as the Power-Law Sensitivity (PLS). This is a function of the observation time, SNR, and the detectors' noise characteristics.

    Description:
    This function calculates the sensitivity of a pair of detectors or a network of detectors to a Gravitational Wave Background (GWB) signal, based on the given parameters. It evaluates the effective noise power spectral density and energy density for the signal and computes the corresponding sensitivity. The analysis can handle both predefined detector networks (such as LISA or ET) and custom detectors with user-supplied parameters.

    - For predefined detector networks (like 'ET triangular' or 'LISA'), the function calculates the sensitivity by considering the relevant response functions and the associated noise power spectral densities.
    - For custom detectors, the function interpolates the supplied PSDs and calculates the overlap response between the detectors.
    - The result is the sensitivity of the system to the GWB signal, given in terms of the Power-Law Sensitivity (PLS), which quantifies the detector network's ability to detect gravitational waves with the specified polarization, frequency range, and observation time.
    '''

    
    def S_eff(orf, Ni, Nj):
        '''Effective noise power spectral density'''
        return np.sqrt(Ni * Nj) / np.abs(orf)

    def Omega_eff(f, orf, Ni, Nj):
        '''Effective energy density'''
        return 10 * np.pi**2 / (3 * (H0**2) / (h**2)) * f**3 * S_eff(orf, Ni, Nj)

    def Omega_beta(f, fref, snr, Tobs, beta, orf, Ni, Nj):
        '''Energy density for a given beta'''
        Tobs = Tobs * 365 * 24 * 3600
        Omega_eff_num = Omega_eff(f, orf, Ni, Nj)
        integrand = (((f / fref) ** beta) / Omega_eff_num) ** 2
        integral = np.trapezoid(integrand, f)
        return snr / np.sqrt(2 * Tobs) / np.sqrt(integral)

    def Omega_GW(f, fref, snr, Tobs, beta, orf, Ni, Nj):
        '''Power spectral density of the GW signal'''
        return Omega_beta(f, fref, snr, Tobs, beta, orf, Ni, Nj) * ((f / fref) ** beta)

    def all_Omega_GW(f, fref, snr, Tobs, orf, Ni, Nj):
        '''Power spectral density of the GW signal for beta values from beta_min to beta_max'''
        beta = np.linspace(-40, 40, 1000)
        Omega = np.array([Omega_GW(f, fref, snr, Tobs, b, orf, Ni, Nj) for b in beta])
        return beta, Omega
    
    # Check if det1 and det2 are predefined network names ('ET triangular' or 'LISA')
    if det1 == 'ET triangular' and det2 == 'Network':
        if fI is None and PnI is None:
            fI, PnI = detectors.detector_Pn('ET X')
        
        if fJ is None and PnJ is None:
            fJ, PnJ = detectors.detector_Pn('ET X')

        psd_A = psd_E = psd_T = np.interp(f, fI, PnI)

        R_AA = Response.overlap('ET A', 'ET A', f, psi, pol)
        R_EE = Response.overlap('ET E', 'ET E', f, psi, pol)
        R_TT = Response.overlap('ET T', 'ET T', f, psi, pol)

        _, Omega_A = all_Omega_GW(f, fref, snr, Tobs, R_AA, psd_A, psd_A)
        _, Omega_E = all_Omega_GW(f, fref, snr, Tobs, R_EE, psd_E, psd_E)
        _, Omega_T = all_Omega_GW(f, fref, snr, Tobs, R_TT, psd_T, psd_T)

        pls_A = np.max(Omega_A, axis=0)
        pls_E = np.max(Omega_E, axis=0)
        pls_T = np.max(Omega_T, axis=0)

        pls = np.array([pls_A, pls_E, pls_T])

        return np.sum(1 / pls**2, axis=0)**(-0.5)

    elif det1 == 'LISA' and det2 == 'Network':
        psd_A = LISA_noise_AET(f, 'A')
        psd_E = LISA_noise_AET(f, 'E')
        psd_T = LISA_noise_AET(f, 'T')

        R_AA = Response.overlap('LISA A', 'LISA A', f, psi, pol)
        R_EE = Response.overlap('LISA E', 'LISA E', f, psi, pol)
        R_TT = Response.overlap('LISA T', 'LISA T', f, psi, pol)

        _, Omega_A = all_Omega_GW(f, fref, snr, Tobs, R_AA, psd_A, psd_A)
        _, Omega_E = all_Omega_GW(f, fref, snr, Tobs, R_EE, psd_E, psd_E)
        _, Omega_T = all_Omega_GW(f, fref, snr, Tobs, R_TT, psd_T, psd_T)

        pls_A = np.max(Omega_A, axis=0)
        pls_E = np.max(Omega_E, axis=0)
        pls_T = np.max(Omega_T, axis=0)

        pls = np.array([pls_A, pls_E, pls_T])

        return np.sum(1 / pls**2, axis=0)**(-0.5)

    elif isinstance(det1, str or list) and isinstance(det2, str or list):
        # Handle the case of custom detectors
        if fI is None and PnI is None:
            fI, PnI = detectors.detector_Pn(det1)
        
        if fJ is None and PnJ is None:
            fJ, PnJ = detectors.detector_Pn(det2)

        PnI = np.interp(f, fI, PnI)
        PnJ = np.interp(f, fJ, PnJ)
        
        orfIJ = overlap.Response.overlap(det1, det2, f, psi, pol, shift_angle, Stokes_parameter)
        beta, Omega = all_Omega_GW(f, fref, snr, Tobs, orfIJ, PnI, PnJ)

        pls = np.max(Omega, axis=0)
        return pls
    
    else:
        raise ValueError('Unknown detectors or networks')




def PLS_2pol(det1, det2, det3, f, fref, pol, snr, Tobs, beta_min, beta_max, psi, shift_angle, fI=None, PnI=None, fJ=None, PnJ=None, fK=None, PnK=None):

    '''
    det1, det2, det3: detectors (string)
    f: frequency array (array float)
    fref: reference frequency (float)
    pol: polarization mode (string)
    snr: signal-to-noise ratio threshold (float)
    Tobs: observation time (float) - in YEARS
    beta_min, beta_max: range of beta values (float)
    shift_angle: shift angle (None or float)

    return: PLS array
    '''

    if fI is None and PnI is None:
        fI, PnI = detectors.detector_Pn(det1)
    if fJ is None and PnJ is None:
        fJ, PnJ = detectors.detector_Pn(det2)
    if fK is None and PnK is None:
        fK, PnK = detectors.detector_Pn(det3)

    PnI = np.interp(f, fI, PnI)
    PnJ = np.interp(f, fJ, PnJ)
    PnK = np.interp(f, fK, PnK)


    orf_12_t = overlap.Response.overlap(det1, det2, f, psi , 't', shift_angle )
    orf_13_x = overlap.Response.overlap(det1, det3, f, psi , pol, shift_angle )
    orf_13_t = overlap.Response.overlap(det1, det3, f, psi , 't', shift_angle )
    orf_12_x = overlap.Response.overlap(det1, det2, f, psi , pol, shift_angle )

    orfIJK = orf_12_t * orf_13_x - orf_13_t * orf_12_x

    def S_eff(orf_12_t, orf_13_x, orf_13_t, orf_12_x,  Ni, Nj, Nk):
        return ((orf_12_t * orf_13_x - orf_13_t * orf_12_x)**2 / (orf_12_t**2 * Ni * Nk + orf_13_t**2 * Ni * Nj))**(-0.5)
    
    def Omega_eff(f, fref, snr, Tobs, orf_12_t, orf_13_x, orf_13_t, orf_12_x, Ni, Nj, Nk):
        return 10* np.pi**2 /(3* (H0**2)/(h**2))* f**3 * S_eff(orf_12_t, orf_13_x, orf_13_t, orf_12_x, Ni, Nj, Nk)
    
    def Omega_beta(f, fref, snr, Tobs, beta, orf_12_t, orf_13_x, orf_13_t, orf_12_x, Ni, Nj, Nk):
        Tobs = Tobs * 365 * 24 * 3600
        Omega_eff_num = Omega_eff(f, fref, snr, Tobs, orf_12_t, orf_13_x, orf_13_t, orf_12_x, Ni, Nj, Nk)
        integrand = (((f/fref)**(beta)) / (Omega_eff_num))**2
        integral = np.trapezoid(integrand, f)
        return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
    
    def Omega_GW(f, fref, snr, Tobs, beta, orf_12_t, orf_13_x, orf_13_t, orf_12_x, Ni, Nj, Nk):
        return Omega_beta(f, fref, snr, Tobs, beta, orf_12_t, orf_13_x, orf_13_t, orf_12_x, Ni, Nj, Nk) * ((f/fref)**(beta))
    
    def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, orf_12_t, orf_13_x, orf_13_t, orf_12_x, Ni, Nj, Nk):
        beta = np.linspace(beta_min, beta_max, 1000)
        Omega = []
        for i in range(len(beta)):
            Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], orf_12_t, orf_13_x, orf_13_t, orf_12_x, Ni, Nj, Nk))     
        return beta, np.array(Omega)
    
    beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, orf_12_t, orf_13_x, orf_13_t, orf_12_x, PnI, PnJ, PnK)

    pls = np.zeros(len(f))
    for i in range(len(f)):
        pls[i] = np.max(Omega[:,i])

    return pls



def PLS_3pol(det1, det2, det3, f, fref, pol, snr, Tobs, beta_min, beta_max, psi, shift_angle, fI=None, PnI=None, fJ=None, PnJ=None, fK=None, PnK=None):

    '''
    det1, det2, det3: detectors (string)
    f: frequency array (array float)
    fref: reference frequency (float)
    pol: polarization mode (string)
    snr: signal-to-noise ratio threshold (float)
    Tobs: observation time (float) - in YEARS
    beta_min, beta_max: range of beta values (float)
    shift_angle: shift angle (None or float)

    return: PLS array
    '''

    if fI is None and PnI is None:
        fI, PnI = detectors.detector_Pn(det1)
    if fJ is None and PnJ is None:
        fJ, PnJ = detectors.detector_Pn(det2)
    if fK is None and PnK is None:
        fK, PnK = detectors.detector_Pn(det3)

    PnI = np.interp(f, fI, PnI)
    PnJ = np.interp(f, fJ, PnJ)
    PnK = np.interp(f, fK, PnK)


    orf_12_t = overlap.Response.overlap(det1, det2, f, psi , 't', shift_angle )
    orf_23_t = overlap.Response.overlap(det2, det3, f, psi , 't', shift_angle )
    orf_31_t = overlap.Response.overlap(det1, det3, f, psi , 't', shift_angle )

    orf_12_v = overlap.Response.overlap(det1, det2, f, psi , 'v', shift_angle )
    orf_23_v = overlap.Response.overlap(det2, det3, f, psi , 'v', shift_angle )
    orf_31_v = overlap.Response.overlap(det1, det3, f, psi , 'v', shift_angle )

    orf_12_s = overlap.Response.overlap(det1, det2, f, psi , 's', shift_angle )
    orf_23_s = overlap.Response.overlap(det2, det3, f, psi , 's', shift_angle )
    orf_31_s = overlap.Response.overlap(det1, det3, f, psi , 's', shift_angle )

    orfIJK = orf_12_t * ( orf_23_s * orf_31_v - orf_31_s * orf_23_v) + \
                orf_23_t * ( orf_31_s * orf_12_v - orf_12_s * orf_31_v) + \
                orf_31_t * ( orf_12_s * orf_23_v - orf_23_s * orf_12_v)
    
    a_1_t = orf_23_s * orf_31_v - orf_31_s * orf_23_v
    a_2_t = orf_31_s * orf_12_v - orf_12_s * orf_31_v
    a_3_t = orf_12_s * orf_23_v - orf_23_s * orf_12_v

    a_1_v = orf_23_s * orf_31_t - orf_31_s * orf_23_t
    a_2_v = orf_31_s * orf_12_t - orf_12_s * orf_31_t
    a_3_v = orf_12_s * orf_23_t - orf_23_s * orf_12_t

    a_1_s = orf_23_t * orf_31_v - orf_31_t * orf_23_v
    a_2_s = orf_31_t * orf_12_v - orf_12_t * orf_31_v   
    a_3_s = orf_12_t * orf_23_v - orf_23_t * orf_12_v

    a_1 = np.zeros(len(f))
    a_2 = np.zeros(len(f))
    a_3 = np.zeros(len(f))

    if pol == 't':
        a_1 = a_1_t
        a_2 = a_2_t
        a_3 = a_3_t

    elif pol == 'v':
        a_1 = a_1_v
        a_2 = a_2_v
        a_3 = a_3_v

    elif pol == 's':
        a_1 = a_1_s
        a_2 = a_2_s
        a_3 = a_3_s

    else:
        raise ValueError('Unknown polarization')
    

    def S_eff(orfIJK, a_1, a_2, a_3, Ni, Nj, Nk):
        den = a_1**2 * Ni * Nk + a_2**2 * Ni * Nj + a_3**2 * Nj * Nk
        return (orfIJK**2 / den)**(-0.5)
    
    def Omega_eff(f, fref, snr, Tobs, orfIJK, a_1, a_2, a_3, Ni, Nj, Nk):
        return 10* np.pi**2 /(3* (H0**2)/(h**2))* f**3 * S_eff(orfIJK, a_1, a_2, a_3, Ni, Nj, Nk)
    
    def Omega_beta(f, fref, snr, Tobs, beta, orfIJK, a_1, a_2, a_3, Ni, Nj, Nk):
        Tobs = Tobs * 365 * 24 * 3600
        Omega_eff_num = Omega_eff(f, fref, snr, Tobs, orfIJK, a_1, a_2, a_3, Ni, Nj, Nk)
        integrand = (((f/fref)**(beta)) / (Omega_eff_num))**2
        integral = np.trapezoid(integrand, f)
        return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
    
    def Omega_GW(f, fref, snr, Tobs, beta, orfIJK, a_1, a_2, a_3, Ni, Nj, Nk):
        return Omega_beta(f, fref, snr, Tobs, beta, orfIJK, a_1, a_2, a_3, Ni, Nj, Nk) * ((f/fref)**(beta))
    
    def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, orfIJK, a_1, a_2, a_3, Ni, Nj, Nk):
        beta = np.linspace(beta_min, beta_max, 1000)
        Omega = []
        for i in range(len(beta)):
            Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], orfIJK, a_1, a_2, a_3, Ni, Nj, Nk))     
        return beta, np.array(Omega)
    
    beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, orfIJK, a_1, a_2, a_3, PnI, PnJ, PnK)

    pls = np.zeros(len(f))
    for i in range(len(f)):
        pls[i] = np.max(Omega[:,i])

    return pls


