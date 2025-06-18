import numpy as np
from numpy import sin, pi, sqrt
from mpmath import mp
from gwbird import detectors as det
from gwbird.skymap import Basis, AngularPatternFunction
from scipy.special import sph_harm
from gwbird.utils import H0, h, c

'''
The nell module contain the following classes:
    - AngularResponse: Class for the angular response function.
    - Sensitivity_ell: Class for the sensitivity curve for the multipoles.
'''


# angular overlap redunction function - angular response

class AngularResponse:
   
    def R_ell(ell, det1, det2, f, pol, psi, shift_angle=False):
        '''
        Anisotropic response function for gravitational wave detectors or networks.
        # https://arxiv.org/abs/2201.08782

        Parameters:
        - ell: int (Multipole to consider)
        - det1, det2: str or list of str
            The name of the detector(s) to consider.
            The names must be in the list of detectors available in the response module.
            The list of available detectors can be obtained by calling the function detectors.available_detectors().
            The names of the detectors are case sensitive.
            If you want to provide a custom detector, you can provide the following information in a list:

            H = [c, xA, xB, l, name]

            - c: array_like of length 3 (Position of the detector in the Earth-centered frame in meters)
            - xA: array_like of length 3 (Unit vector pointing towards the detector in the Earth-centered frame)
            - xB: array_like of length 3 (Unit vector pointing towards the detector in the Earth-centered frame)
            - l: float (Length of the arm in meters)
            - name: str (Name of the detector)
            
        - f: array_like (Frequency in Hz)
        - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar)
        - psi: float (Polarization angle in radians)
        - shift_angle: bool or float, optional (Shift angle for ET detectors)

        Returns:
        - R_ell: array_like (Angular response function)
        '''

        def Rellm_integrand(ell, m, theta, phi, psi, c1, u1, v1, c2, u2, v2, f, pol, L1, L2):
            
            '''
            Integrand of the anisotropic response function

            Parameters:
            - ell: int (Multipole to consider)
            - m: int (Azimuthal number)
            - theta: array_like (theta in radians [0, pi])
            - phi: array_like (phi in radians [0, 2*pi])
            - psi: float (Polarization angle in radians)
            - c1, u1, v1: array_like (Detector 1 position parameters)
            - c2, u2, v2: array_like (Detector 2 position parameters)
            - f: array_like (Frequency in Hz)
            - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
            - L1: float (Detector 1 arm length)
            - L2: float (Detector 2 arm length)
            
            Returns:
            - integrand: array_like (Integrand of the anisotropic response function)
            '''


            f = f[:, None, None] 
            
            F1 = AngularPatternFunction.F(theta, phi, psi, c1, u1, v1, f, L1)
            F2 = AngularPatternFunction.F(theta, phi, psi, c2, u2, v2, f, L2)

            sph_harm_val = sph_harm(m, ell, phi, theta)

            if pol == 't' or pol == 'I':
                gamma_ij = (5 / (8 * pi)) * (F1[0] * np.conj(F2[0]) + F1[1] * np.conj(F2[1])) * sph_harm_val * sqrt(4 * pi) * sin(theta)
            elif pol == 'v':
                gamma_ij = (5 / (8 * pi)) * (F1[2] * np.conj(F2[2]) + F1[3] * np.conj(F2[3])) * sph_harm_val * sqrt(4 * pi) * sin(theta)
            elif pol == 's':
                k = 0
                xi = 1/3 * ((1+2*k)/(1+k))
                gamma_ij = (xi * 15/(1+2*k)/(4*pi))*(F1[4] * np.conj(F2[4]) +  k*F1[5]*np.conj(F2[5])) * sph_harm_val * sqrt(4 * pi) * sin(theta)
            elif pol == 'V': 
                gamma_ij = 1j*(5/(8*pi))*( F1[0]* np.conj( F2[1]) - F1[1] *np.conj(F2[0])) *  sin(theta) * sph_harm_val * sqrt(4 * pi)
            else:
                raise ValueError('Unknown polarization')
            return gamma_ij

        def Rellm(ell, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L1, L2):
            
            '''
            Integral of the anisotropic response function

            Parameters:
            - ell: int (Multipole to consider)
            - m: int (Azimuthal number)
            - u1, v1, c1: array_like (Detector 1 position parameters)
            - u2, v2, c2: array_like (Detector 2 position parameters)
            - psi: float (Polarization angle in radians in [0, pi])
            - f: array_like (Frequency in Hz)
            - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
            - L1: float

            Returns:
            - Rellm: array_like (Integral of the anisotropic response function)
            '''

            N = 100
            theta = np.linspace(0, np.pi, N)
            phi = np.linspace(0, 2*np.pi, N)
            Theta, Phi = np.meshgrid(theta, phi)
            integrand = Rellm_integrand(ell, m, Theta, Phi, psi, c1, u1, v1, c2, u2, v2, f, pol, L1, L2)

            gamma = np.trapezoid(np.trapezoid(integrand,theta, axis=1), phi.reshape(1, 1, 100))

            real_part = np.array([mp.mpf(x.real) for row in gamma for x in row])
            imag_part = np.array([mp.mpf(x.imag) for row in gamma for x in row])
            real_part = np.array(real_part, dtype=np.longdouble)
            imag_part = np.array(imag_part, dtype=np.longdouble)
    
            return real_part + 1j*imag_part


        def R_ell_func(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi):
            '''    
            Compute the angular response for a pair of detectors

            Parameters:
            - ell: int (Multipole to consider)
            - c1, u1, v1: array_like (Detector 1 position parameters)
            - c2, u2, v2: array_like (Detector 2 position parameters)
            - f: array_like (Frequency in Hz)
            - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
            - L1: float (Detector 1 arm length)
            - L2: float (Detector 2 arm length)
            - psi: float (Polarization angle in radians)

            Returns:
            - R_ell_func: array_like (Angular response for a pair of detectors)
            '''
            m_values = np.arange(-ell, ell+1)
            total = 0
            for m in m_values:
                total += np.abs(Rellm(ell, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L1, L2))**2
            return np.sqrt(total)

        # Handling LISA detectors
        if any(det in det1 for det in ['LISA A', 'LISA E', 'LISA T']) and any(det in det2 for det in ['LISA A', 'LISA E', 'LISA T']):
            c1, u1, v1, L1, _ = det.detector('LISA X', shift_angle=None)
            c2, u2, v2, L2, _ = det.detector('LISA Y', shift_angle=None)

        # Handling ET detectors
        elif any(det in det1 for det in ['ET A', 'ET E', 'ET T']) and any(det in det2 for det in ['ET A', 'ET E', 'ET T']):
            c1, u1, v1, L1, _ = det.detector('ET X', shift_angle=None)
            c2, u2, v2, L2, _ = det.detector('ET Y', shift_angle=None)

        else:
            if isinstance(det1, str):
                c1, u1, v1, L1, _ = det.detector(det1, shift_angle)
            else:
                c1, u1, v1, L1, _ = det1

            if isinstance(det2, str):
                c2, u2, v2, L2, _ = det.detector(det2, shift_angle)
            else:
                c2, u2, v2, L2, _ = det2

            return R_ell_func(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi)

        # Define response functions for even multipoles (l%2 == 0)
        # AET basis handling 
        if ell % 2 == 0:
            def R_AA_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi):
                '''
                Compute the angular response for the AA channel in the AET basis

                Parameters:
                - ell: int (Multipole to consider)
                - c1, u1, v1: array_like (Detector 1 position parameters)
                - c2, u2, v2: array_like (Detector 2 position parameters)
                - f: array_like (Frequency in Hz)   
                - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
                - L1: float (Detector 1 arm length)
                - L2: float (Detector 2 arm length)
                - psi: float (Polarization angle in radians)

                Returns:
                - R_AA_ell: array_like (Angular response for the AA channel in the AET basis)
                '''
                total = 0
                for m in np.arange(-ell, ell+1):
                    total += np.abs((1 + np.exp(-4j*np.pi*m/3)) * Rellm(ell, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L1, L2)
                                    - 2 * Rellm(ell, m, u1, v1, c1, u2, v2, c2, psi, f, pol, 1, L2))**2
                return np.sqrt(total/4)

            def R_TT_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi):
                '''
                Compute the angular response for the TT channel in the AET basis

                Parameters:
                - ell: int (Multipole to consider)
                - c1, u1, v1: array_like (Detector 1 position parameters)
                - c2, u2, v2: array_like (Detector 2 position parameters)
                - f: array_like (Frequency in Hz)
                - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
                - L1: float (Detector 1 arm length)
                - L2: float (Detector 2 arm length)
                - psi: float (Polarization angle in radians)

                Returns:
                - R_TT_ell: array_like (Angular response for the TT channel in the AET basis)
                '''
                total = 0
                for m in np.arange(-ell, ell+1):
                    total += (1 + 2*np.cos(2*np.pi*m/3))**2 * np.abs(Rellm(ell, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L1, L2)
                                                                    + 2 * Rellm(ell, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L1, L2))**2
                return np.sqrt(total/9)

            def R_AE_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi):
                '''
                Compute the angular response for the AE channel in the AET basis

                Parameters:
                - ell: int (Multipole to consider)
                - c1, u1, v1: array_like (Detector 1 position parameters)
                - c2, u2, v2: array_like (Detector 2 position parameters)
                - f: array_like (Frequency in Hz)
                - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
                - L1: float (Detector 1 arm length)
                - L2: float (Detector 2 arm length)
                - psi: float (Polarization angle in radians)

                Returns:
                - R_AE_ell: array_like (Angular response for the AE channel in the AET basis)
                '''
                total = 0
                for m in np.arange(-ell, ell+1):
                    total += np.sin(np.pi*m/3)**2 * np.abs((1 + np.exp(2j*np.pi*m/3)) * Rellm(ell, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L1, L2)
                                                            - 2 * Rellm(ell, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L1, L2))**2
                return np.sqrt(total/3)
                              
            def R_AT_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi):
                '''
                Compute the angular response for the AT channel in the AET basis

                Parameters:
                - ell: int (Multipole to consider)
                - c1, u1, v1: array_like (Detector 1 position parameters)
                - c2, u2, v2: array_like (Detector 2 position parameters)
                - f: array_like (Frequency in Hz)
                - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
                - L1: float (Detector 1 arm length)
                - L2: float (Detector 2 arm length)
                - psi: float (Polarization angle in radians)

                Returns:
                - R_AT_ell: array_like (Angular response for the AT channel in the AET basis)
                '''
                total = 0
                for m in np.arange(-ell, ell+1):
                    total += np.sin(np.pi*m/3)**2 * np.abs((1 + np.exp(2j*np.pi*m/3)) * Rellm(ell, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L1, L2)
                                                            + Rellm(ell, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L1, L2))**2
                return np.sqrt(2*total/3)

            if (det1.endswith('A') and det2.endswith('A')) or (det1.endswith('E') and det2.endswith('E')):
                return R_AA_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi)
            elif det1.endswith('T') and det2.endswith('T'):
                return R_TT_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi)
            elif (det1.endswith('A') and det2.endswith('E')) or (det1.endswith('E') and det2.endswith('A')):
                return R_AE_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi)
            elif (det1.endswith('A') and det2.endswith('T')) or (det1.endswith('T') and det2.endswith('A')) or (det1.endswith('E') and det2.endswith('T')) or (det1.endswith('T') and det2.endswith('E')):
                return R_AT_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi)
            else:
                raise ValueError('Unknown combination of detectors')

        else:
            def R_AA_ell(f):
                '''
                Compute the angular response for the AA channel in the AET basis

                Parameters:
                - f: array_like (Frequency in Hz)

                Returns:
                - R_AA_ell: array_like (Angular response for the AA channel in the AET basis)
                '''
                return np.zeros(len(f))
            
            def R_AE_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi):
                '''
                Compute the angular response for the AE channel in the AET basis

                Parameters:
                - ell: int (Multipole to consider)
                - c1, u1, v1: array_like (Detector 1 position parameters)
                - c2, u2, v2: array_like (Detector 2 position parameters)
                - f: array_like (Frequency in Hz)
                - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
                - L1: float (Detector 1 arm length)
                - L2: float (Detector 2 arm length)
                - psi: float (Polarization angle in radians)

                Returns:
                - R_AE_ell: array_like (Angular response for the AE channel in the AET basis)
                '''
                m_values = np.arange(-ell, ell+1)
                total = 0
                for m in m_values:
                    total += (1 + 2*np.cos(2*np.pi*m/3))**2 * (np.abs(Rellm(ell, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L1, L2)))**2
                return np.sqrt(total/3)
            
            def R_AT_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi):
                '''
                Compute the angular response for the AT channel in the AET basis

                Parameters:
                - ell: int (Multipole to consider)
                - c1, u1, v1: array_like (Detector 1 position parameters)
                - c2, u2, v2: array_like (Detector 2 position parameters)
                - f: array_like (Frequency in Hz)
                - pol: str (Polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
                - L1: float (Detector 1 arm length)
                - L2: float (Detector 2 arm length)
                - psi: float (Polarization angle in radians)

                Returns:
                - R_AT_ell: array_like (Angular response for the AT channel in the AET basis)
                '''
                m_values = np.arange(-ell, ell+1)
                total = 0 
                for m in m_values:
                    total += sin(np.pi*m/3)**2 * (np.abs(Rellm(ell, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L1, L2) ))**2
                return np.sqrt(2*total)

            if (det1.endswith('A') and det2.endswith('A')) or (det1.endswith('E') and det2.endswith('E')):
                return R_AA_ell(f)
            elif (det1.endswith('A') and det2.endswith('E')) or (det1.endswith('E') and det2.endswith('A')):
                return R_AE_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi)
            elif (det1.endswith('A') and det2.endswith('T')) or (det1.endswith('T') and det2.endswith('A')) or (det1.endswith('E') and det2.endswith('T')) or (det1.endswith('T') and det2.endswith('E')):
                return R_AT_ell(ell, c1, u1, v1, c2, u2, v2, f, pol, L1, L2, psi)
            elif det1.endswith('T') and det2.endswith('T'):
                return R_AA_ell(f)
            else:
                raise ValueError('Unknown combination of channels')

    def R_ell_pairwise(ell, f, pi, pj, Di, Dj, pol, psi=0):

        '''
        Compute the angular response for a pair of pulsars

        Parameters:
        - ell: int (multipole to consider)
        - pi: len(3) array_like (pulsar i position in the xyz coordinates)
        - pj: len(3) array_like (pulsar j position in the xyz coordinates)
        - f: array_like (frequency in Hz)
        - pol: str (polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
        - psi: float, optional (Polarization angle in radians (default is 0))

        Returns:
        - angular_response: array_like (angular response for a pair of pulsar)
        '''



        def Rellm_integrand_PTA(ell, m, theta, phi, psi, f, pi, pj, Di, Dj, pol):
            '''
            Compute the integrand of the angular response function for a pair of pulsar

            Parameters:
            - ell: int (multipole to consider)
            - m: int 
            - theta: array_like (Polar angle in [0, pi])
            - phi: array_like (Azimuthal angle in [0, 2*pi])
            - psi: float (Polarization angle in [0, pi])
            - f: array_like (Frequency in Hz)
            - pi: array_like (Position of the first pulsar)
            - pj: array_like (Position of the second pulsar)
            - Di: float (Distance to the first pulsar)
            - Dj: float (Distance to the second pulsar)
            - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar breathing, 'l' for scalar longitudinal, 'I' for intensity, 'V' for circular)

            Returns:
            - gamma_ij: array_like (Integrand of the angular overlap reduction function for a pair of pulsar)

            '''

            Omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]
            Fi = AngularPatternFunction.F_pulsar(theta, phi, psi, pi)
            Fj = AngularPatternFunction.F_pulsar(theta, phi, psi, pj)

            f = f[:, None, None]  # shape (Nf, 1, 1)

            f = f / 1e-9  
            Di = Di / 1e19  
            Dj = Dj / 1e19
            Di = np.asarray(Di, dtype=np.longdouble)
            Dj = np.asarray(Dj, dtype=np.longdouble)
            c = 3e8 / 1e8  #

            fDc_i  = f * Di / c *1e2
            fDc_j = f * Dj / c * 1e2

            phase_i = 2 * np.pi * fDc_i * (1 + np.einsum('iab,i->ab', Omega, pi)) 
            phase_i = np.asarray(phase_i, dtype=np.longdouble)  
            phase_j = 2 * np.pi * fDc_j * (1 + np.einsum('iab,i->ab', Omega, pj))
            phase_j = np.asarray(phase_j, dtype=np.longdouble)

            exp1 = 1 - np.exp(-1j*phase_i) 
            exp2 = 1 - np.exp(1j*phase_j)

            pulsarterms = np.asarray(exp1 * exp2, dtype=np.float64)

            if pol=='t' or pol == 'I':
                gamma_ij = 3* (Fi[0] * Fj[0] + Fi[1] * Fj[1])  *  sph_harm(m, ell, phi, theta)* np.sqrt(4* np.pi)/ (8*np.pi) * pulsarterms
            elif pol=='v':
                gamma_ij = 3* (Fi[2] * Fj[2] + Fi[3] * Fj[3])  *  sph_harm(m, ell, phi, theta)* np.sqrt(4* np.pi)/ (8*np.pi) * pulsarterms
            elif pol=='s':
                gamma_ij = 3* (Fi[4] * Fj[4] )  *  sph_harm(m, ell, phi, theta)* np.sqrt(4* np.pi)/ (8*np.pi) * pulsarterms
            elif pol=='l':
                gamma_ij = 3* (Fi[5] * Fj[5]) *  sph_harm(m, ell, phi, theta)* np.sqrt(4* np.pi)/ (8*np.pi) * pulsarterms
            elif pol=='V':
                gamma_ij = 3j* ((Fi[0] * Fj[1] - Fi[1] * Fj[0]) ) * sph_harm(m, ell, phi, theta)* np.sqrt(4* np.pi)/ (8*np.pi) *  pulsarterms
            else:
                raise ValueError('Unknown polarization')            
            return gamma_ij 

        def Rellm_PTA(ell, m, pi, pj, Di, Dj, f, pol, psi):
            '''
            Compute the integral of the angular response function for a pair of pulsar

            Parameters:
            - ell: int (multipole to consider)
            - m: int (azimuthal number)
            - pi: len(3) array_like (pulsar i position in the xyz coordinates)
            - pj: len(3) array_like (pulsar j position in the xyz coordinates)
            - Di: float (distance to the first pulsar)
            - Dj: float (distance to the second pulsar)
            - f: array_like (frequency in Hz)
            - pol: str (polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
            - psi: float (polarization angle in radians)
            
            Returns:
            - gamma_ellm: array_like (integral of the angular response function for a pair of pulsar)
            '''

            N = 200
            eps = 1e-8
            theta = np.linspace(eps, np.pi- eps, N)
            phi = np.linspace(eps, 2*np.pi-eps, N)
            Theta, Phi = np.meshgrid(theta, phi) 
            integrand = Rellm_integrand_PTA(ell, m, Theta, Phi, psi, f, pi, pj, Di, Dj, pol)
            integral = np.trapezoid(np.trapezoid(np.sin(Theta) * integrand, theta), phi)
            return integral
        
        # uncomment this line to use the Rellm_PTA function
            # (you will have to add the m value by hand in Rellm_PTA function)
        # m = your_m_value
        #return Rellm_PTA(ell,m,  pi, pj, Di, Dj, psi,f, pol)

        # comment the following lines to use the Rellm_PTA function
        
        def Rell_func_PTA(ell, f, pi, pj, Di, Dj, pol, psi):
            '''
            Compute the angular response for a pair of pulsar

            Parameters:
            - ell: int (multipole to consider)
            - f: array_like (frequency in Hz)
            - pi: len(3) array_like (pulsar i position in the xyz coordinates)
            - pj: len(3) array_like (pulsar j position in the xyz coordinates)
            - Di: float (distance to the first pulsar)
            - Dj: float (distance to the second pulsar)
            - pol: str (polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
            - psi: float (polarization angle in radians)

            Returns:
            - gamma_ell: array_like (angular response for a pair of pulsar)
            '''
            # if ell==0:
            #     return np.abs(np.real(Rellm_PTA(0, 0, pi, pj, Di, Dj, psi, f, pol)))
            
            # else:
            #     gamma_l = 0
            #     for m in range(-ell, ell+1):
            #         gamma_l += np.abs(Rellm_PTA(ell, m, pi, pj, Di, Dj, psi, f, pol))**2
            #     return np.sqrt(gamma_l) 

            gamma_l = 0
            for m in range(-ell, ell+1):
                gamma_l += np.abs(Rellm_PTA(ell, m, pi, pj, Di, Dj, f, pol, psi))**2
            return np.sqrt(gamma_l) 

        return Rell_func_PTA(ell, f, pi, pj, Di, Dj, pol, psi)   
    

  
    def R_ell_PTA(ell, f, pol, psi=0):

        '''
        Compute the overlap reduction function for a set of pulsars

        Parameters:
        - ell: int (multipole to consider)
        - f: array_like (frequency in Hz)
        - pol: str (polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)
        - psi: float, optional (Polarization angle in radians (default is 0))

        Returns:
        - angular response:  array_like (angular response for a set of pulsars)
        '''

        N, p, D = det.get_NANOGrav_pulsars()
        
        angular_response = np.zeros(len(f))

        for i in range(N):
            for j in range(i+1, N):
                angular_response += AngularResponse.R_ell_pairwise(ell, f, p[i], p[j],  D[i],  D[j], pol, psi)
                
        return angular_response 


# Sensitivity curve for the multipoles

class Sensitivity_ell:


    def APLS_ell(det1, det2, ell, f, pol, psi, fref, snr, Tobs, Cl, shift_angle=None, fI=None, PnI=None, fJ=None, PnJ=None):
        """
        Computes PLS_ell for detectors, handling both individual detectors and networks.

        # https://arxiv.org/pdf/2201.08782 eq.4.42 - 4.43

        Parameters:
        - det1 (str): Name of the first detector ( also 'LISA', 'ET', or custom detector).
        - det2 (str): Name of the second detector ('Network' or custom detector).
        - ell (int): Multipole moment.
        - f (array): Frequency array.
        - pol (float): Polarization.
        - psi (float): Polarization angle.
        - fref (float): Reference frequency.
        - snr (float): Signal-to-noise ratio threshold.
        - Tobs (float): Observation time in years.
        - Cl (float): Cl parameter for multipole.
        - shift_angle (float, optional): Shift angle.
        - fI, PnI, fJ, PnJ (array, optional): Frequency and noise power spectral density for custom detectors.
        
        Returns:
        - pls (array_like): power law sensitivity curve  (h^2 \Omega_{GW}(f))
        """

        def Omega_eff_ell(det1, det2, Rl, f, fI=None, PnI=None, fJ=None, PnJ=None):
            """
            Computes the effective Omega_eff_ell for two given detectors.

            Parameters:
            - det1 (str): Name of the first detector.
            - det2 (str): Name of the second detector.
            - Rl (array): Angular response function.
            - f (array): Frequency array.
            - fI, PnI, fJ, PnJ (array, optional): Frequency and noise power spectral density for custom detectors.

            Returns:
            - array: Effective Omega_eff_ell.
            """

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
            """
            Computes Omega_beta.
            
            Parameters:
            - f (array): Frequency array.
            - fref (float): Reference frequency.
            - snr (float): Signal-to-noise ratio threshold.
            - Tobs (float): Observation time in years.
            - beta (float): Beta parameter.
            - Omega_eff_l (array): Effective Omega_eff_ell.
            
            Returns:
            - array: Omega_beta.
            """

            Tobs_sec = Tobs * 365 * 24 * 3600
            integrand = (((f/fref)**beta) / Omega_eff_l)**2 * Cl
            integral = np.trapezoid(integrand, f)
            return snr / np.sqrt(2 * Tobs_sec) / np.sqrt(integral)

        def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
            """
            Computes Omega_GW.
            
            Parameters:
            - f (array): Frequency array.
            - fref (float): Reference frequency.
            - snr (float): Signal-to-noise ratio threshold.
            - Tobs (float): Observation time in years.
            - beta (float): Beta parameter.
            - Omega_eff_l (array): Effective Omega_eff_ell.

            Returns:
            - array: Omega_GW.
            """
            return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * (f/fref)**beta

        def all_Omega_GW(f, fref, snr, Tobs, Omega_eff_l):
            """
            Computes all Omega_GW curves for different beta values.
            
            Parameters:
            - f (array): Frequency array.
            - fref (float): Reference frequency.
            - snr (float): Signal-to-noise ratio threshold.
            - Tobs (float): Observation time in years.
            - Omega_eff_l (array): Effective Omega_eff_ell.

            Returns:
            - array: All Omega_GW curves.
            """

            beta = np.linspace(-40, 40, 1000) # beta_min = -40, beta_max = 40
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
    

    
    
    def APLS_ell_PTA(ell, f, snr, Tobs, Cl, pol, psi):

        '''
        Compute the power law sensitivity curve for NANOGrav pulsars catalog

        Parameters:
        - ell: int (multipole to consider)
        - f: array_like (frequency in Hz)
        - snr: float (signal to noise ratio)
        - Tobs: float (observation time in years)
        - Cl: float (Cl parameter for multipole)
        - pol: str (polarization: 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular polarization)

        Returns:
        - pls: array_like (power law sensitivity curve  (h^2 \Omega_{GW}(f))
        '''

        N, p, D = det.get_NANOGrav_pulsars()
        
        def PTA_Pn():
            DT = (365*24*3600)/20 # s
            s = 100 * 1e-9 #s
            return 2* (s**2) * DT

        def PTA_Sn(f):
            f = np.asarray(f) # Ensure f is a NumPy array
            mask = f>= 1/(365*24*60*60*Tobs)
            return np.where(mask, PTA_Pn() * 12 * (np.pi**2) * f**2, 1) # Apply the mask to the result
        
        def PTA_Omegaeff_all(ell, f, p, d, pol, psi):
            '''
            Returns the effective energy density of the PTA
            '''
            s = 0
            for i in range(N):
                for j in range(i+1, N): 
                    s +=  AngularResponse.R_ell_pairwise(ell, p[i], p[j], D[i], D[j], f, pol, psi)**2 

            return 2 * np.pi * np.pi * f**3 / np.sqrt(s/(PTA_Sn(f)* PTA_Sn(f))) / (3* ((H0/h)**2))
        
        Omega_eff = PTA_Omegaeff_all(ell, f, p, d, pol, psi)

        def Omega_beta_PTA(f, snr, Tobs, Cl, beta, Omega_eff): 
            Tobs = Tobs*365*24*3600
            fref = 1e-8
            integrand = ((f/fref)**(2*beta))/ (Omega_eff**2) * Cl
            integral = np.trapezoid(integrand, f)
            return snr / np.sqrt(2*Tobs*integral)

        def Omega_GW_PTA(f, beta, fref, snr, Tobs, Cl, Omega_eff):
            return Omega_beta_PTA(f, snr, Tobs, Cl, beta, Omega_eff) * ((f/fref)**(beta))

        def all_Omega_GW_PTA(f, snr, Tobs, Cl, Omega_eff):
            beta = np.linspace(-8, 8, 50)
            fref = 1e-8
            Omega = []
            for i in range(len(beta)):
                Omega.append(Omega_GW_PTA(f, beta[i], fref, snr, Tobs, Cl, Omega_eff))     
            return beta, np.array(Omega)
        


        beta, Omega = all_Omega_GW_PTA(f, snr, Tobs, Cl, Omega_eff)
        pls = np.zeros(len(f))
        for i in range(len(f)):
            pls[i] = np.max(Omega[:,i])
        return pls

