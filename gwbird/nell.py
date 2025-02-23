import numpy as np
from numpy import cos, sin, pi, sqrt
from mpmath import mp
from gwbird import detectors as det
from gwbird.skymap import AngularPatternFunction
from scipy.special import sph_harm
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

        gamma_x = np.trapz(f_values, x_values.reshape(1, 100, 1), axis=1)
        gamma = np.trapz(gamma_x, y_values.reshape(1, 1, 100))

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
            integral = np.trapz(integrand, f)
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



    # def PLS_ell(det1, det2, ell, f, pol, psi, fref, snr, Tobs, Cl, shift_angle=None, fI=None, PnI=None, fJ=None, PnJ=None): # PLS_ell

    #     '''
    #     Parameters:

    #     - det1, det2: str or list of str
    #         The name of the detector(s) to consider.
    #         The names must be in the list of detectors available in the response module.
    #         The list of available detectors can be obtained by calling the function detectors.available_detectors().
    #         The names of the detectors are case sensitive.
    #         If you want to provide a custom detector, you can provide the following information in a list:

    #         H = [c, xA, xB, l, name]

    #         - c: array_like of length 3 (Position of the detector in the Earth-centered frame in meters)
    #         - xA: array_like of length 3 (Unit vector pointing towards the detector in the Earth-centered frame)
    #         - xB: array_like of length 3 (Unit vector pointing towards the detector in the Earth-centered frame)
    #         - l: float (Length of the arm in meters)
    #         - name: str (Name of the detector)

    #     - Rl: array_like (Angular response for the multipole ell)
    #     - f = array_like (Frequency in Hz)
    #     - fref = float (Reference frequency in Hz)
    #     - snr = float (Signal-to-noise ratio threshold)
    #     - Tobs = float (Observation time in years)
    #     - beta_min = float (Minimum tilt to consider)
    #     - beta_max = float (Maximum tilt to consider)
    #     - Cl = float (Cl for the multipole ell)

    #     Optional parameters:
    #     - fI = bool or array_like (Frequency in Hz for the detector I)
    #     - PnI = bool or array_like (Power spectral density for the detector I)
    #     - fJ = bool or array_like (Frequency in Hz for the detector J)
    #     - PnJ = bool or array_like (Power spectral density for the detector J)

    #     '''

    #     def Omega_eff_ell(det1, det2, Rl, f, fI=None, PnI=None, fJ=None, PnJ=None): # N_ell
    #         '''
    #         det1, det2: detectors (string)
    #         Rl: anisotropic response function (array float)
    #         f: frequency array (array float)
    #         fI, PnI, fJ, PnJ: frequency and noise power spectral density arrays
    #         '''
        
    #         if fI is not None and PnI is not None and fJ is not None and PnJ is not None:
    #             Pni = np.interp(f, fI, PnI)
    #             Pnj = np.interp(f, fJ, PnJ)
    #         else:
    #             fi, PnI = det.detector_Pn(det1)
    #             fj, PnJ = det.detector_Pn(det2)
    #             Pni = np.interp(f, fi, PnI)
    #             Pnj = np.interp(f, fj, PnJ)
            
    #         Nl = 10 * np.pi**2 * np.sqrt(4*np.pi)  / (3* (H0/h)**2) * f**3 * np.sqrt(Pni * Pnj) / Rl
            
    #         return Nl
        
    #     def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
    #         Tobs = Tobs * 365 * 24 * 3600
    #         integrand = (((f/fref)**(beta)) / (Omega_eff_l))**2 * Cl
    #         integral = np.trapz(integrand, f)
    #         return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
        
    #     def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
    #         return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * ((f/fref)**(beta))
        
    #     def all_Omega_GW(f, fref, snr, Tobs,  Omega_eff_l):
    #         beta = np.linspace(-40, 40, 1000)
    #         Omega = []
    #         for i in range(len(beta)):
    #             Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], Omega_eff_l))     
    #         return beta, np.array(Omega)

    #     if det1 == 'LISA' and det2 == 'Network':
    #         if ell == 0:
    #             R_AA = AngularResponse.R_ell(ell, 'LISA A', 'LISA A', f, pol, psi, shift_angle)
    #             R_TT = AngularResponse.R_ell(ell, 'LISA T', 'LISA T', f, pol, psi, shift_angle)

    #             psd_A = det.LISA_noise_AET(f, 'A')
    #             psd_E = det.LISA_noise_AET(f, 'E')
    #             psd_T = det.LISA_noise_AET(f, 'T')

    #             Omega_eff_AA = Omega_eff_ell('LISA A', 'LISA A', R_AA, f, f, psd_A, f, psd_A)/np.sqrt(4*np.pi)
    #             Omega_eff_EE = Omega_eff_ell('LISA E', 'LISA E', R_AA, f, f, psd_E, f, psd_E)/np.sqrt(4*np.pi)
    #             Omega_eff_TT = Omega_eff_ell('LISA T', 'LISA T', R_TT, f, f, psd_T, f, psd_T)/np.sqrt(4*np.pi)

    #             Omega_eff = np.array([Omega_eff_AA, Omega_eff_EE, Omega_eff_TT])

    #         elif ell % 2 == 0 and ell != 0:
    #             R_AA = AngularResponse.R_ell(ell, 'LISA A', 'LISA A', f, pol, psi, shift_angle)
    #             R_EE = R_AA
    #             R_TT = AngularResponse.R_ell(ell, 'LISA T', 'LISA T', f, pol, psi, shift_angle)
    #             R_AE = AngularResponse.R_ell(ell, 'LISA A', 'LISA E', f, pol, psi, shift_angle)
    #             R_AT = AngularResponse.R_ell(ell, 'LISA A', 'LISA T', f, pol, psi, shift_angle)
    #             R_ET = R_AT

    #             psd_A = det.LISA_noise_AET(f, 'A')
    #             psd_E = det.LISA_noise_AET(f, 'E')
    #             psd_T = det.LISA_noise_AET(f, 'T')

    #             Omega_eff_AA = Omega_eff_ell('LISA A', 'LISA A', R_AA, f, f, psd_A, f, psd_A)/np.sqrt(4*np.pi)
    #             Omega_eff_EE = Omega_eff_ell('LISA E', 'LISA E', R_EE, f, f, psd_E, f, psd_E)/np.sqrt(4*np.pi)
    #             Omega_eff_TT = Omega_eff_ell('LISA T', 'LISA T', R_TT, f, f, psd_T, f, psd_T)/np.sqrt(4*np.pi)
    #             Omega_eff_AE = Omega_eff_ell('LISA A', 'LISA E', R_AE, f, f, psd_A, f, psd_E)/np.sqrt(4*np.pi)
    #             Omega_eff_AT = Omega_eff_ell('LISA A', 'LISA T', R_AT, f, f, psd_A, f, psd_T)/np.sqrt(4*np.pi)
    #             Omega_eff_ET = Omega_eff_ell('LISA E', 'LISA T', R_ET, f, f, psd_E, f, psd_T)/np.sqrt(4*np.pi)

    #             Omega_eff = np.array([Omega_eff_AA, Omega_eff_EE, Omega_eff_TT, Omega_eff_AE, Omega_eff_AT, Omega_eff_ET])

    #         elif ell % 2 != 0 and ell != 0:
    #             R_AE = AngularResponse.R_ell(ell, 'LISA A', 'LISA E', f, pol, psi, shift_angle)
    #             R_AT = AngularResponse.R_ell(ell, 'LISA A', 'LISA T', f, pol, psi, shift_angle)
    #             R_ET = R_AT

    #             psd_A = det.LISA_noise_AET(f, 'A')
    #             psd_E = det.LISA_noise_AET(f, 'E')
    #             psd_T = det.LISA_noise_AET(f, 'T')

    #             Omega_eff_AE = Omega_eff_ell('LISA A', 'LISA E', R_AE, f, f, psd_A, f, psd_E)/np.sqrt(4*np.pi)
    #             Omega_eff_AT = Omega_eff_ell('LISA A', 'LISA T', R_AT, f, f, psd_A, f, psd_T)/np.sqrt(4*np.pi)
    #             Omega_eff_ET = Omega_eff_ell('LISA E', 'LISA T', R_ET, f, f, psd_E, f, psd_T)/np.sqrt(4*np.pi)

    #             Omega_eff = np.array([Omega_eff_AE, Omega_eff_AT, Omega_eff_ET])

    #         else:
    #             raise ValueError('Insert a valid multipole')
            
    #         pls_l = np.zeros((len(Omega_eff), len(f)))
    #         for i in range(len(Omega_eff[:,0])):
    #             beta, Omega = all_Omega_GW(f, fref, snr, Tobs, Omega_eff[i,:])
    #             for j in range(len(f)):
    #                 pls_l[i, j] = np.max(Omega[:,j])

    #         return np.sum(1/(pls_l)**2, axis=0)**(-0.5)



    #     else:
    #         Rl = AngularResponse.R_ell(ell, det1, det2, f, pol, psi, shift_angle)

    #         Omega_eff_l = Omega_eff_ell(det1, det2, Rl, f, fI, PnI, fJ, PnJ)/np.sqrt(4*np.pi)

    #         beta, Omega = all_Omega_GW(f, fref, snr, Tobs, Omega_eff_l)

    #         pls_l = np.zeros(len(f))

    #         for i in range(len(f)):
    #             pls_l[i] = np.max(Omega[:,i])

    #         return pls_l
    



    def PLS_l_LISA(f, l, pol, fref, snr, Tobs, beta_min, beta_max, Cl, psi):

        '''
        Parameters:
        - f = array_like (Frequency in Hz)
        - Rl: array_like (Angular response for the multipole ell)
        - fref = float (Reference frequency in Hz)
        - snr = float (Signal-to-noise ratio threshold)
        - Tobs = float (Observation time in years)
        - beta_min = float (Minimum tilt to consider)
        - beta_max = float (Maximum tilt to consider)
        - Cl = float (Cl for the multipole ell)
        - psi = float (Polarization angle in radians)
        
        Return: array float (PLS for the multipole ell)
        '''
        if l == 0:

            Rl_AA = AngularResponse.R_ell_AET(l, 'AA', pol, psi, f)
            Rl_EE = Rl_AA
            Rl_TT =  AngularResponse.R_ell_AET(l, 'TT', pol, psi, f)

            psd_A = det.LISA_noise_AET(f, 'A')
            psd_E = det.LISA_noise_AET(f, 'E')
            psd_T = det.LISA_noise_AET(f, 'T')

            Nl_AA = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_A) / Rl_AA
            Nl_EE = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_E) / Rl_EE
            Nl_TT = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_T * psd_T) / Rl_TT

            Nl = np.array([Nl_AA, Nl_EE, Nl_TT])


            def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
                Tobs = Tobs * 365 * 24 * 3600
                integrand = (((f/fref)**(beta)) / (Omega_eff_l))**2 * Cl
                integral = np.trapz(integrand, f)
                return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
            
            def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
                return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * ((f/fref)**(beta))
            
            def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l):
                beta = np.linspace(beta_min, beta_max, 1000)
                Omega = []
                for i in range(len(beta)):
                    Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], Omega_eff_l))     
                return beta, np.array(Omega)
            
            pls_l = np.zeros((len(Nl), len(f)))
            for i in range(len(Nl[:,0])):
                beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Nl[i,:])
                for j in range(len(f)):
                    pls_l[i, j] = np.max(Omega[:,j])

            
            return np.sum(1/(pls_l)**2, axis=0)**(-0.5)


        elif l % 2 == 0 and l != 0:

            Rl_AA = AngularResponse.R_ell_AET(l, 'AA', pol, psi, f)
            Rl_EE = Rl_AA
            Rl_TT =  AngularResponse.R_ell_AET(l, 'TT', pol, psi, f)
            Rl_AE =  AngularResponse.R_ell_AET(l, 'AE', pol, psi, f)
            Rl_AT =  AngularResponse.R_ell_AET(l, 'AT', pol, psi, f)
            Rl_ET = Rl_AT

            psd_A = det.LISA_noise_AET(f, 'A')
            psd_E = det.LISA_noise_AET(f, 'E')
            psd_T = det.LISA_noise_AET(f, 'T')

            Nl_AA = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_A) / Rl_AA
            Nl_EE = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_E) / Rl_EE
            Nl_TT = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_T * psd_T) / Rl_TT
            Nl_AE = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_E) / Rl_AE
            Nl_AT = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_T) / Rl_AT
            Nl_ET = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_T) / Rl_ET

            Nl = np.array([Nl_AA, Nl_EE, Nl_TT, Nl_AE, Nl_AT, Nl_ET])


            def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
                Tobs = Tobs * 365 * 24 * 3600
                integrand = (((f/fref)**(beta)) / (Omega_eff_l))**2 * Cl
                integral = np.trapz(integrand, f)
                return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
            
            def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
                return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * ((f/fref)**(beta))
            
            def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l):
                beta = np.linspace(beta_min, beta_max, 1000)
                Omega = []
                for i in range(len(beta)):
                    Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], Omega_eff_l))     
                return beta, np.array(Omega)
            
            pls_l = np.zeros((len(Nl), len(f)))
            for i in range(len(Nl[:,0])):
                beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Nl[i,:])
                for j in range(len(f)):
                    pls_l[i, j] = np.max(Omega[:,j])

            
            return np.sum(1/(pls_l)**2, axis=0)**(-0.5)

        else:
            
            Rl_AE =  AngularResponse.R_ell_AET(l, 'AE', pol, psi, f)
            Rl_AT =  AngularResponse.R_ell_AET(l, 'AT', pol, psi, f)
            Rl_ET = Rl_AT

            psd_A = det.LISA_noise_AET(f, 'A')
            psd_E = det.LISA_noise_AET(f, 'E')
            psd_T = det.LISA_noise_AET(f, 'T')

            Nl_AE = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_E) / Rl_AE
            Nl_AT = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_T) / Rl_AT
            Nl_ET = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_T) / Rl_ET

            Nl = np.array([Nl_AE, Nl_AT, Nl_ET])


            def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
                Tobs = Tobs * 365 * 24 * 3600
                integrand = (((f/fref)**(beta)) / (Omega_eff_l))**2 * Cl
                integral = np.trapz(integrand, f)
                return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
            
            def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
                return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * ((f/fref)**(beta))
            
            def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l):
                beta = np.linspace(beta_min, beta_max, 1000)
                Omega = []
                for i in range(len(beta)):
                    Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], Omega_eff_l))     
                return beta, np.array(Omega)
            
            pls_l = np.zeros((len(Nl), len(f)))
            for i in range(len(Nl[:,0])):
                beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Nl[i,:])
                for j in range(len(f)):
                    pls_l[i, j] = np.max(Omega[:,j])

            
            return np.sum(1/(pls_l)**2, axis=0)**(-0.5)








    

