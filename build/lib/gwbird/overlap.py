import numpy as np
import mpmath as mp
from numpy import sin, pi
from gwbird import detectors 
from gwbird.skymap import Basis, AngularPatternFunction
from gwbird.utils import c
mp.dps = 50


'''
The overlap module contain the following classes and functions:
    - Response: Calculate the overlap response between two detectors/ pulsars network
'''

class Response:

    # detectors

    def overlap(det1, det2, f, pol, psi, shift_angle=None):
        """
        Calculate the overlap response between two detectors.

        R = Response.overlap(det1, det2, f, psi, pol, shift_angle=False)

        Parameters:
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
        - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular)
        - psi: float (Polarization angle in radians)
        - shift_angle: float (Shift the angle of the response if considering ET 2L in radians)

        Return:
        - overlap: array_like (Overlap reduction function between the two detectors)

        """

        def overlap_integrand(theta, phi, psi, c1, xA1, xB1, l1, c2, xA2, xB2, l2, f, pol):
            
            '''
            Integrand of the overlap reduction function

            Parameters:
            - theta: array_like (Polar angle in radians)
            - phi: array_like (Azimuthal angle in radians)
            - psi: float (Polarization angle in radians)
            - c1: array_like (Position of the first detector in meters)
            - xA1: array_like (Unit vector pointing towards the first detector first arm)
            - xB1: array_like (Unit vector pointing towards the first detector second arm)
            - l1: float (Length of the arm of the first detector in meters)
            - c2: array_like (Position of the second detector in meters)
            - xA2: array_like (Unit vector pointing towards the second detector first arm)
            - xB2: array_like (Unit vector pointing towards the second detector second arm)
            - l2: float (Length of the arm of the second detector in meters)
            - f: array_like (Frequency in Hz)
            - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular)

            Return:
            - integrand: array_like (Integrand of the overlap reduction function)
            '''

            F1 = AngularPatternFunction.F(theta, phi, psi, c1, xA1, xB1, f, l1)
            F2 = AngularPatternFunction.F(theta, phi, psi, c2, xA2, xB2, f, l2)

            f = f[:, None, None] 

            if pol == 't' or pol=='I': # https://arxiv.org/pdf/1310.5300
                gamma_ij = (5/(8*pi))*( F1[0]* np.conj( F2[0]) + F1[1] *np.conj(F2[1])) *sin(theta)
            elif pol == 'v': # https://arxiv.org/pdf/2105.13197
                gamma_ij = (5/(8*pi))*( F1[2]* np.conj( F2[2]) + F1[3] *np.conj(F2[3])) *sin(theta)
            elif pol == 's': # https://arxiv.org/pdf/2105.13197
                k = 0
                xi = 1/3 * ((1+2*k)/(1+k))
                gamma_ij = xi * 15/(1+2*k)/(4*pi)*( F1[4]* np.conj( F2[4]) + k*F1[5]*np.conj(F2[5]) ) * sin(theta)
            elif pol=='V': # https://arxiv.org/pdf/0707.0535
                gamma_ij = 1j*(5/(8*pi))*( F1[0]* np.conj( F2[1]) - F1[1] *np.conj(F2[0])) *sin(theta)
            else:
                raise ValueError('Unknown polarization')
            return gamma_ij
            

        def overlap_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, pol):
            
            '''
            Overlap reduction function for a given polarization

            Parameters:
            - xA1: array_like (Unit vector pointing towards the first detector first arm)
            - xB1: array_like (Unit vector pointing towards the first detector second arm)
            - c1: array_like (Position of the first detector in meters)
            - l1: float (Length of the arm of the first detector in meters)
            - xA2: array_like (Unit vector pointing towards the second detector first arm)
            - xB2: array_like (Unit vector pointing towards the second detector second arm)
            - c2: array_like (Position of the second detector in meters)
            - l2: float (Length of the arm of the second detector in meters)
            - psi: float (Polarization angle in radians)
            - f: array_like (Frequency in Hz)
            - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular)

            Return:
            - gamma: array_like (Overlap reduction function)
            '''
            N = 100
            theta = np.linspace(0, np.pi, N)
            phi = np.linspace(0, 2*np.pi, N)
            Theta, Phi = np.meshgrid(theta, phi)
            integrand = overlap_integrand(Theta, Phi, psi, c1, xA1, xB1, l1, c2, xA2, xB2, l2, f, pol) # gamma values
            integral = np.trapezoid( np.trapezoid(integrand, theta, axis=1), phi)
            return np.real(integral)
    

        def get_detector_params(det):
            if isinstance(det, str):
                # Handle standard detectors
                if det in ['LISA A', 'LISA E', 'LISA T']:
                    cX, xAX, xBX, lX, _ = detectors.detector('LISA X', shift_angle=None)
                    cY, xAY, xBY, lY, _ = detectors.detector('LISA Y', shift_angle=None)
                    cZ, xAZ, xBZ, lZ, _ = detectors.detector('LISA Z', shift_angle=None)
                    
                    if det == 'LISA A':
                        return cX, xAX, xBX, lX, det
                    elif det == 'LISA E':
                        return cY, xAY, xBY, lY, det
                    elif det == 'LISA T':
                        return cZ, xAZ, xBZ, lZ, det
                elif det in ['ET A', 'ET E', 'ET T']:
                    cX, xAX, xBX, lX, _ = detectors.detector('ET X', shift_angle=None)
                    cY, xAY, xBY, lY, _ = detectors.detector('ET Y', shift_angle=None)
                    cZ, xAZ, xBZ, lZ, _ = detectors.detector('ET Z', shift_angle=None)
                    
                    if det == 'ET A':
                        return cX, xAX, xBX, lX, det
                    elif det == 'ET E':
                        return cY, xAY, xBY, lY, det
                    elif det == 'ET T':
                        return cZ, xAZ, xBZ, lZ, det
                else:
                    return detectors.detector(det, shift_angle)
            elif isinstance(det, list) and len(det) == 5:
                # Handle custom detectors
                c, xA, xB, l, name = det
                return np.array(c), np.array(xA), np.array(xB), l, name
            else:
                raise ValueError(f"Invalid detector format: {det}")

        # Handle standard detectors or custom detectors
        c1, xA1, xB1, l1, name1 = get_detector_params(det1)
        c2, xA2, xB2, l2, name2 = get_detector_params(det2)

        # Check if detectors are lists or strings and convert them to appropriate string names for the special_map
        if isinstance(det1, list):
            name1 = det1[4]  # Custom detectors have a name in the 5th element
        if isinstance(det2, list):
            name2 = det2[4]  # Custom detectors have a name in the 5th element

        # Special handling for LISA and ET cases
        special_map = {
            ('LISA A', 'LISA A'): ('LISA X', 'LISA Y', -1.),
            ('LISA E', 'LISA E'): ('LISA X', 'LISA Y', -1.),
            ('LISA T', 'LISA T'): ('LISA X', 'LISA Y', 2.),
            ('ET A', 'ET A'): ('ET X', 'ET Y', -1),
            ('ET E', 'ET E'): ('ET X', 'ET Y', -1),
            ('ET T', 'ET T'): ('ET X', 'ET Y', 2)
        }

        # Check if det1 and det2 match any special combinations
        if (name1, name2) in special_map:
            det_x, det_y, factor = special_map[(name1, name2)]
            c1, xA1, xB1, l1, _ = detectors.detector(det_x, shift_angle=None)
            c2, xA2, xB2, l2, _ = detectors.detector(det_y, shift_angle=None)
            auto = overlap_func(xA1, xB1, c1, l1, xA1, xB1, c1, l1, psi, f, pol)
            cross = overlap_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, pol)
            return auto + factor * cross

        # Check for invalid combinations
        special_channels = {'LISA A', 'LISA E', 'LISA T', 'ET A', 'ET E', 'ET T'}
        if (name1 in special_channels or name2 in special_channels) and (name1 != name2):
            raise ValueError("Put a valid combination of channels")
        
        
        # General case    
        return np.real(overlap_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, pol))
    

    # pulsar timing array


    def overlap_pairwise(f, pi, pj, Di, Dj, pol, psi):

        '''
        Compute the overlap reduction function between two pulsars

        Parameters:
        - f: array_like (Frequency in Hz)
        - pi: len(3) array_like (Position of the first pulsar)
        - pj: len(3) array_like (Position of the second pulsar)
        - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar breathing, 'l' for scalar longitudinal, 'I' for intensity, 'V' for circular)
        - psi: float (Polarization angle in radians)

        Return:
        - overlap: array_like (Overlap reduction function between two pulsars)
        '''

        def gamma_integrand(theta, phi, psi, f, pi, pj, Di, Dj, pol):

            '''
            Integrand of the overlap reduction function for two pulsars

            Parameters:
            - theta: array_like (Polar angle in [0, pi])
            - phi: array_like (Azimuthal angle in [0, 2*pi])
            - psi: float (Polarization angle in [0, pi])
            - f: array_like (Frequency in Hz)
            - pi: array_like (Position of the first pulsar)
            - pj: array_like (Position of the second pulsar)
            - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar breathing, 'l' for scalar longitudinal, 'I' for intensity, 'V' for circular)

            Return:
            - gamma_ij: array_like (Integrand of the overlap reduction function)

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

            phase_i = -2 * np.pi * fDc_i * (1 + np.einsum('iab,i->ab', Omega, pi)) 
            phase_i = np.asarray(phase_i, dtype=np.longdouble)  
            phase_j = 2 * np.pi * fDc_j * (1 + np.einsum('iab,i->ab', Omega, pj))
            phase_j = np.asarray(phase_j, dtype=np.longdouble)

            expi = 1 - np.exp(1j*phase_i) 
            expj = 1 - np.exp(1j*phase_j)

            pulsarterms = np.real(expi * expj)#.astype(np.float64)

            if pol == 't' or pol == 'I':
                gamma_ij = 3 * (Fi[0] * Fj[0] + Fi[1] * Fj[1]) * (1/(8*np.pi)) * np.sin(theta) * pulsarterms 
            elif pol == 'v':
                gamma_ij = 3 * (Fi[2] * Fj[2]+ Fi[3] * Fj[3]) * (1/(8*np.pi)) * np.sin(theta) * pulsarterms
            elif pol == 's': # scalar breathing
                gamma_ij = 3 * Fi[4] * Fj[4] * (1/(8*np.pi)) * np.sin(theta) * pulsarterms 
            elif pol == 'l': # scalar longitudinal
                gamma_ij = 3 * Fi[5] * np.conj(Fj[5]) * (1/(8*np.pi)) * np.sin(theta) * pulsarterms
            elif pol == 'V':
                gamma_ij = 3j* (Fi[0] * Fj[1] - Fi[1] * Fj[0]) * (1/(8*np.pi)) * np.sin(theta)* pulsarterms 
            else:
                raise ValueError('Unknown polarization')
            return gamma_ij

        def gamma(pi, pj, Di, Dj, f, pol, psi):
            '''
            Overlap reduction function between two pulsars

            Parameters:
            - pi: array_like (Position of the first pulsar)
            - pj: array_like (Position of the second pulsar)
            - Di: float (Distance to the first pulsar)
            - Dj: float (Distance to the second pulsar)
            - f: array_like (Frequency in Hz)
            - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar breathing, 'l' for scalar longitudinal, 'I' for intensity, 'V' for circular)
            - psi: float (Polarization angle in radians, default is 0)

            Return:
            - integral: array_like (Overlap reduction function between two pulsars)

            '''

            N = 200
            epsilon = 0#1e-8
            theta = np.linspace(epsilon, np.pi- epsilon, N)
            phi = np.linspace(epsilon, 2*np.pi- epsilon, N)
            Theta, Phi = np.meshgrid(theta, phi) 
            integrand = np.real(gamma_integrand(Theta, Phi, psi, f, pi, pj, Di, Dj, pol)) 
            integral = np.trapezoid(np.trapezoid(integrand, phi, axis=1), theta, axis=1)
            return np.real(integral)

        return gamma(pi, pj, Di, Dj, f, pol, psi)
    

    def overlap_PTA(f, pol, psi):

        '''
        Compute the overlap reduction function for a set of pulsars (NANOGrav)

        Parameters:
        - f: array_like (Frequency in Hz)
        - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar breathing, 'l' for scalar longitudinal, 'I' for intensity, 'V' for circular)
        - psi: float (Polarization angle in radians)

        Return:
        - overlap: array_like (overlap reduction function for a set of pulsars)
        '''

        N, p, D = detectors.get_NANOGrav_pulsars()
        
        overlap = np.zeros(len(f))


        for i in range(N):
            for j in range(i+1, N):
                overlap += Response.overlap_pairwise(f, p[i], p[j], D[i], D[j], pol, psi)
                
        return overlap
    
