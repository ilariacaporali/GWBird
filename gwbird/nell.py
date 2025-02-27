import numpy as np
from numpy import cos, sin, pi, sqrt
from mpmath import mp
from gwbird import detectors as det
from gwbird.skymap import AngularPatternFunction
from scipy.special import sph_harm, sph_harm_y
from gwbird.utils import c, H0, h


# angular overlap redunction function - angular response

class AngularResponse:



    def Rellm_integrand(l, m, x, y, psi, c1, u1, v1, c2, u2, v2, c, f, pol, L):
        
        '''
        Integrand of the anisotropic response function
        # refs: Bartolo et al. 2022 
        '''

        f = f.reshape(len(f), 1, 1)
        
        F1 = AngularPatternFunction.F(x, y, psi, c1, u1, v1, f, L)
        F2 = AngularPatternFunction.F(x, y, psi, c2, u2, v2, f, L)

        sph_harm_val = sph_harm(m, l, y, x)

        if pol == 't':
            return (5 / (8 * pi)) * (F1[0] * np.conj(F2[0]) + F1[1] * np.conj(F2[1])) * sph_harm_val * sqrt(4 * pi) * sin(x)
        elif pol == 'v':
            return (5 / (8 * pi)) * (F1[2] * np.conj(F2[2]) + F1[3] * np.conj(F2[3])) * sph_harm_val * sqrt(4 * pi) * sin(x)
        elif pol == 's':
            return (15 / (4 * pi)) * (F1[4] * np.conj(F2[4])) * sph_harm_val * sqrt(4 * pi) * sin(x)
        elif pol == 'I':
            return (5 / (8 * pi)) * (F1[0] * np.conj(F2[0]) + F1[1] * np.conj(F2[1])) * sph_harm_val * sqrt(4 * pi) * sin(x)
        elif pol == 'V':
            return 1j*(5/(8*pi))*( F1[0]* np.conj( F2[1]) - F1[1] *np.conj(F2[0])) * sph_harm_val * sqrt(4 * pi) * sin(x)
        else:
            raise ValueError('Unknown polarization')

    def Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L):
        
        '''
        Integral of the anisotropic response function
        '''

        x_values = np.linspace(0, pi, 100)
        y_values = np.linspace(0, 2*pi, 100)
        X, Y = np.meshgrid(x_values,y_values) 

        f_values = AngularResponse.Rellm_integrand(l, m, X, Y, psi, c1, u1, v1, c2, u2, v2, c, f, pol, L)

        gamma_x = np.trapezoid(f_values, x_values.reshape(1, 100, 1), axis=1)
        gamma = np.trapezoid(gamma_x, y_values.reshape(1, 1, 100))

        real_part = np.array([mp.mpf(x.real) for row in gamma for x in row])
        imag_part = np.array([mp.mpf(x.imag) for row in gamma for x in row])
        real_part = np.array(real_part, dtype=np.float64)
        imag_part = np.array(imag_part, dtype=np.float64)
   
        return real_part + 1j*imag_part

    def R_ell(l, det1, det2, f, pol, psi, shift_angle=False):
        '''
        Anisotropic response function for gravitational wave detectors or networks.

        Parameters:
        - l: int (Multipole to consider)
        - det1, det2: str or list of str (Detector names)
        - f: array_like (Frequency in Hz)
        - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar)
        - psi: float (Polarization angle in radians)
        - shift_angle: bool or float, optional (Shift angle for ET detectors)

        Returns:
        - R_ell: array_like (Angular response function)
        '''

        # Function to calculate the anisotropic response for custom detectors
        def R_ell_func(l, c1, u1, v1, c2, u2, v2, f, pol, L, psi):
            m_values = np.arange(-l, l+1)
            total = 0
            for m in m_values:
                total += np.abs(AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L))**2
            return np.sqrt(total)

        # Handling LISA detectors
        if any(det in det1 for det in ['LISA A', 'LISA E', 'LISA T']) and any(det in det2 for det in ['LISA A', 'LISA E', 'LISA T']):
            c1, u1, v1, L, _ = det.detector('LISA X', shift_angle=None)
            c2, u2, v2, L, _ = det.detector('LISA Y', shift_angle=None)

        # Handling ET detectors
        elif any(det in det1 for det in ['ET A', 'ET E', 'ET T']) and any(det in det2 for det in ['ET A', 'ET E', 'ET T']):
            c1, u1, v1, L, _ = det.detector('ET X', shift_angle=None)
            c2, u2, v2, L, _ = det.detector('ET Y', shift_angle=None)

        else:
            if isinstance(det1, str):
                c1, u1, v1, L, _ = det.detector(det1, shift_angle)
            else:
                c1, u1, v1, L, _ = det1

            if isinstance(det2, str):
                c2, u2, v2, L, _ = det.detector(det2, shift_angle)
            else:
                c2, u2, v2, L, _ = det2

            return R_ell_func(l, c1, u1, v1, c2, u2, v2, f, pol, L, psi)

        # Define response functions for even multipoles (l%2 == 0)
        if l % 2 == 0:
            def R_AA_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi):
                total = 0
                for m in np.arange(-l, l+1):
                    total += np.abs((1 + np.exp(-4j*np.pi*m/3)) * AngularResponse.Rellm(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                    - 2 * AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L))**2
                return np.sqrt(total/4)

            def R_TT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi):
                total = 0
                for m in np.arange(-l, l+1):
                    total += (1 + 2*np.cos(2*np.pi*m/3))**2 * np.abs(AngularResponse.Rellm(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                                                    + 2 * AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L))**2
                return np.sqrt(total/9)

            def R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi):
                total = 0
                for m in np.arange(-l, l+1):
                    total += np.sin(np.pi*m/3)**2 * np.abs((1 + np.exp(2j*np.pi*m/3)) * AngularResponse.Rellm(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                                            - 2 * AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L))**2
                return np.sqrt(total/3)
                              
            def R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi):
                    total = 0
                    for m in np.arange(-l, l+1):
                        total += np.sin(np.pi*m/3)**2 * np.abs((1 + np.exp(2j*np.pi*m/3)) * AngularResponse.Rellm(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                                                + AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L))**2
                    return np.sqrt(2*total/3)

            if (det1.endswith('A') and det2.endswith('A')) or (det1.endswith('E') and det2.endswith('E')):
                return R_AA_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi)
            elif det1.endswith('T') and det2.endswith('T'):
                return R_TT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi)
            elif (det1.endswith('A') and det2.endswith('E')) or (det1.endswith('E') and det2.endswith('A')):
                return R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi)
            elif (det1.endswith('A') and det2.endswith('T')) or (det1.endswith('T') and det2.endswith('A')) or (det1.endswith('E') and det2.endswith('T')) or (det1.endswith('T') and det2.endswith('E')):
                return R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi)
            else:
                raise ValueError('Unknown combination of detectors')

        else:
            def R_AA_ell(f):
                return np.zeros(len(f))
            
            def R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi):
                m_values = np.arange(-l, l+1)
                total = 0
                for m in m_values:
                    total += (1 + 2*np.cos(2*np.pi*m/3))**2 * (np.abs(AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L)))**2
                return np.sqrt(total/3)
            
            def R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi):
                m_values = np.arange(-l, l+1)
                total = 0 
                for m in m_values:
                    total += sin(np.pi*m/3)**2 * (np.abs(AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L) ))**2
                return np.sqrt(2*total)

            if (det1.endswith('A') and det2.endswith('A')) or (det1.endswith('E') and det2.endswith('E')):
                return R_AA_ell(f)
            elif (det1.endswith('A') and det2.endswith('E')) or (det1.endswith('E') and det2.endswith('A')):
                return R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi)
            elif (det1.endswith('A') and det2.endswith('T')) or (det1.endswith('T') and det2.endswith('A')) or (det1.endswith('E') and det2.endswith('T')) or (det1.endswith('T') and det2.endswith('E')):
                return R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L, psi)
            elif det1.endswith('T') and det2.endswith('T'):
                return R_AA_ell(f)
            else:
                raise ValueError('Unknown combination of detectors')

    def R_ell_pairwise(ell, pi, pj, f):

        '''
        Compute the angular response for a pair of pulsars

        parameters:
        ell: multipole
        pi: pulsar 1
        pj: pulsar 2
        f: frequency array

        return:
        angular response: angular response for a pair of pulsars


        '''
        
        def gamma_integrand_ellm(ell, m, theta, phi, psi, p1, p2,):
            Fp1 = AngularPatternFunction.F_pulsar(theta, phi, psi, p1)
            Fp2 = AngularPatternFunction.F_pulsar(theta, phi, psi, p2)
            gamma_ij = 3/ 2 * (Fp1[0] * Fp2[0] + Fp1[1] * Fp2[1])
            return gamma_ij *  sph_harm_y(ell, m, theta, phi)* np.sqrt(4* np.pi)/ (4*np.pi)

        def gamma_ellm(ell, m, p1, p2, f):
            theta = np.linspace(0, np.pi, 100)
            phi = np.linspace(0, 2*np.pi, 100)
            Theta, Phi = np.meshgrid(theta, phi)
            integrand = gamma_integrand_ellm(ell, m, Theta, Phi, 0, p1, p2)
            integral = np.trapezoid(np.trapezoid(np.sin(Theta) * integrand, theta), phi)
            return np.abs(integral)
        
        def gamma_ell(ell, p1, p2, f):
            gamma_l = 0
            for m in range(-ell, ell+1):
                gamma_l += np.abs(gamma_ellm(ell, m, p1, p2, f))**2
            return np.sqrt(gamma_l) 

        return gamma_ell(ell, pi, pj, f)   
    

    def R_ell_EPTA(ell, f):

        '''

        Compute the overlap reduction function for a set of pulsars

        parameters:
        ell: multipole
        f: frequency array

        return:
        angular respone: angular response for a set of pulsars

        '''

        pulsar_xyz, _, _ = det.get_EPTA_pulsars()
        N_pulsar = len(pulsar_xyz)
        
        angular_response = np.zeros(len(f))

        for i in range(N_pulsar):
            for j in range(i +1, N_pulsar):
                angular_response += AngularResponse.R_ell_pairwise(ell, pulsar_xyz[i], pulsar_xyz[j], f)
                
        return angular_response 

    
            
    def R_ell_NANOGrav(ell, f):

        '''

        Compute the overlap reduction function for a set of pulsars

        parameters:
        ell: multipole
        f: frequency array

        return:
        angular respone: angular response for a set of pulsars

        '''

        N_pulsar, pulsar_xyz, DIST_array = det.get_NANOGrav_pulsars()
        
        angular_response = np.zeros(len(f))

        for i in range(N_pulsar):
            for j in range(i +1, N_pulsar):
                angular_response += AngularResponse.R_ell_pairwise(ell, pulsar_xyz[i], pulsar_xyz[j], f)
                
        return angular_response 




class Sensitivity_ell:

    # Bartolo et al. 2022 eq.4.42 - 4.43

    # confrontare con schnell

    def PLS_ell(det1, det2, ell, f, pol, psi, fref, snr, Tobs, Cl, shift_angle=None, fI=None, PnI=None, fJ=None, PnJ=None):
        """
        Computes PLS_ell for LISA, ET, or other detectors, handling both individual detectors and networks.

        Args:
            det1 (str): Name of the first detector ('LISA', 'ET', 'Network', etc.).
            det2 (str): Name of the second detector ('Network' or another detector).
            ell (int): Multipole moment.
            f (array): Frequency array.
            pol (float): Polarization.
            psi (float): Polarization angle.
            shift_angle (float, optional): Shift angle.
            fref (float): Reference frequency.
            snr (float): Signal-to-noise ratio threshold.
            Tobs (float): Observation time in years.
            Cl (float): Cl parameter for multipole.
            fI, PnI, fJ, PnJ (array, optional): Frequency and noise power spectral density for custom detectors.
            
        Returns:
            array: Sensitivity PLS_ell.
        """

        def Omega_eff_ell(det1, det2, Rl, f, fI=None, PnI=None, fJ=None, PnJ=None):
            """Computes the effective Omega_eff_ell for two given detectors."""
            if fI is not None and PnI is not None and fJ is not None and PnJ is not None:
                Pni = np.interp(f, fI, PnI)
                Pnj = np.interp(f, fJ, PnJ)
            else:
                fi, PnI = det.detector_Pn(det1)
                fj, PnJ = det.detector_Pn(det2)
                Pni = np.interp(f, fi, PnI)
                Pnj = np.interp(f, fj, PnJ)
            
            return 10 * np.pi**2 * np.sqrt(4*np.pi) / (3 * (H0/h)**2) * f**3 * np.sqrt(Pni * Pnj) / Rl

        def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
            """Computes Omega_beta."""
            Tobs_sec = Tobs * 365 * 24 * 3600
            integrand = (((f/fref)**beta) / Omega_eff_l)**2 * Cl
            integral = np.trapezoid(integrand, f)
            return snr / np.sqrt(2 * Tobs_sec) / np.sqrt(integral)

        def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
            """Computes Omega_GW."""
            return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * (f/fref)**beta

        def all_Omega_GW(f, fref, snr, Tobs, Omega_eff_l):
            """Computes all Omega_GW curves for different beta values."""
            beta = np.linspace(-40, 40, 1000)
            Omega = np.array([Omega_GW(f, fref, snr, Tobs, b, Omega_eff_l) for b in beta])
            return beta, Omega

        # Define channels for LISA and ET
        channels = ['A', 'E', 'T']

        if det1 == 'LISA' or det1 == 'ET':
            channels = channels
        else:
            channels = None  # General case

        if channels:
            # LISA-Network or ET-Network case
            psd = {ch: det.LISA_noise_AET(f, ch) if det1 == 'LISA' else np.interp(f, *det.detector_Pn('ET X')) for ch in channels}

            if ell == 0:
                pairs = [(ch, ch) for ch in channels]
            elif ell % 2 == 0:
                pairs = [(ch1, ch2) for ch1 in channels for ch2 in channels if ch1 <= ch2]
            else:  # Odd ell
                pairs = [(ch1, ch2) for ch1 in channels for ch2 in channels if ch1 < ch2]

            Omega_eff_list = []
            for ch1, ch2 in pairs:
                R = AngularResponse.R_ell(ell, f"{det1} {ch1}", f"{det1} {ch2}", f, pol, psi, shift_angle)
                Omega_eff = Omega_eff_ell(f"{det1} {ch1}", f"{det1} {ch2}", R, f, f, psd[ch1], f, psd[ch2]) / np.sqrt(4*np.pi)
                Omega_eff_list.append(Omega_eff)

        else:
            # General case: two arbitrary detectors
            R_ell = AngularResponse.R_ell(ell, det1, det2, f, pol, psi, shift_angle)
            Omega_eff_list = [Omega_eff_ell(det1, det2, R_ell, f, fI, PnI, fJ, PnJ) / np.sqrt(4*np.pi)]

        Omega_eff_array = np.array(Omega_eff_list)

        # Compute sensitivity
        pls_l = np.zeros((len(Omega_eff_array), len(f)))
        for i in range(len(Omega_eff_array)):
            beta, Omega = all_Omega_GW(f, fref, snr, Tobs, Omega_eff_array[i])
            pls_l[i, :] = np.max(Omega, axis=0)

        return np.sum(1 / (pls_l) ** 2, axis=0) ** (-0.5)
    

    def apls_PTA_EPTA(ell, f, snr, Tobs, Cl):

        '''
        Compute the power law sensitivity curve for PTA

        parameters:
        ell: multipole, integer
        f: frequency array
        snr: signal to noise ratio
        Tobs: observation time in years
        Cl (float): Cl parameter for multipole.

        return:
        pls: power law sensitivity curve

        '''

        def PTA_Pn(wn, dt):
            return 2 * (wn**2) * dt * 1e-12

        def PTA_Sn(f, wn, dt):
            f = np.asarray(f) # Ensure f is a NumPy array
            mask = f >= 8e-9 # Create a boolean mask where True indicates elements greater than or equal to 8e-9
            return np.where(mask, PTA_Pn(wn, dt) * 12 * (np.pi**2) * f**2, 1) # Apply the mask to the result
        
        def PTA_Omegaeff_all(ell, f, p, wn, dt):
            s = 0
            N = len(p)
            for i in range(N):
                for j in range(i+1, N):
                    s +=  AngularResponse.R_ell_pairwise(ell, p[i], p[j], f)**2 / (PTA_Sn(f, wn[i], dt[i])* PTA_Sn(f, wn[j], dt[j]))

            return 2 * np.pi * np.pi * f**3 / np.sqrt(s) / (3* ((H0/h)**2))
        

        def Omega_beta_PTA(ell, f, snr, Tobs, Cl, beta, p, wn, dt):
            Tobs = Tobs*365*24*3600
            fref = 1e-8
            integrand = ((f/fref)**(2*beta))/ (PTA_Omegaeff_all(ell, f, p, wn, dt)**2) * Cl
            integral = np.trapezoid(integrand, f)
            return snr / np.sqrt(2*Tobs*integral)

        def Omega_GW_PTA(ell, f,  beta, fref, snr, Tobs, Cl,  p, wn, dt):
            return Omega_beta_PTA(ell, f, snr, Tobs, Cl, beta, p, wn, dt) * ((f/fref)**(beta))

        def all_Omega_GW_PTA(ell, f, snr, Tobs, Cl, p, wn, dt):
            beta = np.linspace(-8, 8, 50)
            fref = 1e-8
            Omega = []
            for i in range(len(beta)):
                Omega.append(Omega_GW_PTA(ell, f, beta[i], fref, snr, Tobs, Cl, p, wn, dt))     
            return beta, np.array(Omega)
        
        p, wn, dt = det.get_EPTA_pulsars()
        beta, Omega = all_Omega_GW_PTA(ell, f, snr, Tobs, Cl, p, wn, dt)
        pls = np.zeros(len(f))
        for i in range(len(f)):
            pls[i] = np.max(Omega[:,i])
        return pls
    
    def apls_PTA_NANOGrav(ell, f, snr, Tobs, Cl):

        '''
        Compute the power law sensitivity curve for PTA

        parameters:
        ell: multipole, integer
        f: frequency array
        snr: signal to noise ratio
        Tobs: observation time in years
        Cl (float): Cl parameter for multipole.

        return:
        pls: power law sensitivity curve

        '''

        def PTA_Pn():
            DT = (365*24*3600)/20 # s
            s = 100 * 1e-9 #s
            return 2* (s**2) * DT

        def PTA_Sn(f):
            f = np.asarray(f) # Ensure f is a NumPy array
            mask = f >= 8e-9 # Create a boolean mask where True indicates elements greater than or equal to 8e-9
            return np.where(mask, PTA_Pn() * 12 * (np.pi**2) * f**2, 1) # Apply the mask to the result
        
        def PTA_Omegaeff_all(ell, f, p):
            s = 0
            N = len(p)
            for i in range(N):
                for j in range(i+1, N):
                    s +=  AngularResponse.R_ell_pairwise(ell, p[i], p[j], f)**2 / (PTA_Sn(f)* PTA_Sn(f))

            return 2 * np.pi * np.pi * f**3 / np.sqrt(s) / (3* ((H0/h)**2))
        

        def Omega_beta_PTA(ell, f, snr, Tobs, Cl, beta, p):
            Tobs = Tobs*365*24*3600
            fref = 1e-8
            integrand = ((f/fref)**(2*beta))/ (PTA_Omegaeff_all(ell, f, p)**2) * Cl
            integral = np.trapezoid(integrand, f)
            return snr / np.sqrt(2*Tobs*integral)

        def Omega_GW_PTA(ell, f,  beta, fref, snr, Tobs, Cl,  p):
            return Omega_beta_PTA(ell, f, snr, Tobs, Cl, beta, p) * ((f/fref)**(beta))

        def all_Omega_GW_PTA(ell, f, snr, Tobs, Cl, p):
            beta = np.linspace(-8, 8, 50)
            fref = 1e-8
            Omega = []
            for i in range(len(beta)):
                Omega.append(Omega_GW_PTA(ell, f, beta[i], fref, snr, Tobs, Cl, p))     
            return beta, np.array(Omega)
        
        _, p, _ = det.get_NANOGrav_pulsars()
        beta, Omega = all_Omega_GW_PTA(ell, f, snr, Tobs, Cl, p)
        pls = np.zeros(len(f))
        for i in range(len(f)):
            pls[i] = np.max(Omega[:,i])
        return pls




  






    

