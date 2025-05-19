import numpy as np
from numpy import sin, pi
from gwbird import detectors 
from gwbird.skymap import Basis, AngularPatternFunction
from gwbird.utils import c

from scipy.integrate import simpson
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.interpolate import RectBivariateSpline



# overlap reduction function averaged over the sky

'''
The overlap module contain the following classes and functions:
    - Response: Calculate the overlap response between two detectors/ pulsars network
'''

class Response:

    # detectors

    def overlap(det1, det2, f, psi, pol, shift_angle=None):
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
        - psi: float (Polarization angle in radians)
        - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular)

        Optional parameters:
        - shift_angle: float (Shift the angle of the response if considering ET 2L in radians)

        """

        def R_integrand(x, y, psi, c1, xA1, xB1, l1, c2, xA2, xB2, l2, f, pol):
            
            '''
            Integrand of the overlap reduction function

            Parameters:
            - x: array_like (Polar angle in radians)
            - y: array_like (Azimuthal angle in radians)
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

            F1 = AngularPatternFunction.F(x, y, psi, c1, xA1, xB1, f, l1)
            F2 = AngularPatternFunction.F(x, y, psi, c2, xA2, xB2, f, l2)

            f = f.reshape(len(f), 1, 1)

            if (pol == 't'): # https://arxiv.org/pdf/1310.5300
                return (5/(8*pi))*\
                    ( F1[0]* np.conj( F2[0]) \
                    + F1[1] *np.conj(F2[1])) \
                    *sin(x)
            
            elif (pol == 'v'): # https://arxiv.org/pdf/2105.13197
                return (5/(8*pi))*\
                    ( F1[2]* np.conj( F2[2]) \
                    + F1[3] *np.conj(F2[3])) \
                    *sin(x)
            
            elif (pol == 's'): # https://arxiv.org/pdf/2105.13197
                return (15/(4*pi))*\
                    ( F1[4]* np.conj( F2[4]) ) \
                    * sin(x)
            
            elif (pol =='I'): # https://arxiv.org/pdf/0707.0535
                return (5/(8*pi))*\
                    ( F1[0]* np.conj( F2[0]) \
                    + F1[1] *np.conj(F2[1])) \
                    *sin(x)
            
            elif(pol=='V'): # https://arxiv.org/pdf/0707.0535
                return -1j*(5/(8*pi))*\
                    ( F1[0]* np.conj( F2[1]) \
                    - F1[1] *np.conj(F2[0])) \
                    *sin(x)
        
            else:
                raise ValueError('Unknown polarization')
            




        def R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, pol):
            
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

            x_values = np.linspace(0, pi, 100)
            y_values = np.linspace(0, 2*pi, 100)
            X, Y = np.meshgrid(x_values,y_values)  
            f_values = R_integrand(X, Y, psi, c1, xA1, xB1, l1, c2, xA2, xB2, l2, f, pol) # gamma values
            gamma_x = np.trapezoid(f_values, x_values, axis=1)
            gamma = np.trapezoid(gamma_x, y_values)
            return np.real(gamma) 
    


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
            ('LISA A', 'LISA A'): ('LISA X', 'LISA Y', -1),
            ('LISA E', 'LISA E'): ('LISA X', 'LISA Y', -1),
            ('LISA T', 'LISA T'): ('LISA X', 'LISA Y', 2),
            ('ET A', 'ET A'): ('ET X', 'ET Y', -1),
            ('ET E', 'ET E'): ('ET X', 'ET Y', -1),
            ('ET T', 'ET T'): ('ET X', 'ET Y', 2)
        }

        # Check if det1 and det2 match any special combinations
        if (name1, name2) in special_map:
            det_x, det_y, factor = special_map[(name1, name2)]
            c1, xA1, xB1, l1, _ = detectors.detector(det_x, shift_angle=None)
            c2, xA2, xB2, l2, _ = detectors.detector(det_y, shift_angle=None)
            auto = R_func(xA1, xB1, c1, l1, xA1, xB1, c1, l1, psi, f, pol)
            cross = R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, pol)
            return auto + factor * cross

        # Check for invalid combinations
        special_channels = {'LISA A', 'LISA E', 'LISA T', 'ET A', 'ET E', 'ET T'}
        if (name1 in special_channels or name2 in special_channels) and (name1 != name2):
            raise ValueError("Put a valid combination of channels")
        
        
        # General case    
        return np.array(R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, pol))
    

    # pulsar timing array


    def pairwise_overlap(f, pi, pj, Di, Dj, pol, psi=0):

        '''
        Compute the overlap reduction function between two pulsars

        Parameters:
        - f: array_like (Frequency in Hz)
        - pi: array_like (Position of the first pulsar)
        - pj: array_like (Position of the second pulsar)
        - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar breathing, 'l' for scalar longitudinal, 'I' for intensity, 'V' for circular)

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
            - pi: array_like (Position of the first pulsar)
            - pj: array_like (Position of the second pulsar)
            - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar breathing, 'l' for scalar longitudinal, 'I' for intensity, 'V' for circular)

            Returns:
            - gamma_ij: array_like (Integrand of the overlap reduction function)

            '''
            Omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]#.T

            Fp1 = AngularPatternFunction.F_pulsar(theta, phi, psi, pi)
            Fp2 = AngularPatternFunction.F_pulsar(theta, phi, psi, pj)

            Omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]

            f = f.reshape(len(f), 1, 1)

            exp1 = (1-np.exp(-2j*np.pi*f*Di*(1+(np.einsum('iab,i->ab', Omega, pi)))/c))
            exp2 = (1-np.exp(2j*np.pi*f*Dj*(1+(np.einsum('iab,i->ab', Omega, pj)))/c))
            

            if pol == 't':
                gamma_ij = 3 * (Fp1[0] * np.conj(Fp2[0]) + Fp1[1] * np.conj(Fp2[1])) * (1/(8*np.pi)) * np.sin(theta) * exp1 * exp2
                return gamma_ij
            elif pol == 'v':
                gamma_ij = 3 * (Fp1[2] * np.conj(Fp2[2]) + Fp1[3] * np.conj(Fp2[3])) * (1/(8*np.pi)) * np.sin(theta) * exp1 * exp2
                return gamma_ij
            elif pol == 's':
                gamma_ij = 3 * Fp1[4] * np.conj(Fp2[4]) * (1/(8*np.pi)) * np.sin(theta) * exp1 * exp2
                return gamma_ij
            elif pol == 'l':
                gamma_ij = 3 * Fp1[5] * np.conj(Fp2[5]) * (1/(8*np.pi)) * np.sin(theta) * exp1 * exp2
                return gamma_ij
            elif pol == 'I':
                gamma_ij = 3 * (Fp1[0] * np.conj(Fp2[0]) + Fp1[1] * np.conj(Fp2[1])) * (1/(8*np.pi)) * np.sin(theta) * exp1 * exp2
                return gamma_ij
            elif pol == 'V':
                gamma_ij = 3j * (Fp1[0] * np.conj(Fp2[1]) - Fp1[1] * np.conj(Fp2[0])) * (1/(8*np.pi)) * np.sin(theta) * exp1 * exp2
                return gamma_ij
            else:
                raise ValueError('Unknown polarization')

        def gamma(pi, pj, Di, Dj, f, pol, psi):
            '''
            Overlap reduction function between two pulsars

            Parameters:
            - pi: array_like (Position of the first pulsar)
            - pj: array_like (Position of the second pulsar)
            - f: array_like (Frequency in Hz)
            - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar breathing, 'l' for scalar longitudinal, 'I' for intensity, 'V' for circular)

            Returns:
            - integral: array_like (Overlap reduction function between two pulsars)

            '''
        
            theta = np.linspace(0, np.pi, 400)
            phi = np.linspace(0, 2*np.pi, 400)
            Theta, Phi = np.meshgrid(theta, phi) 

            integrand = gamma_integrand(Theta, Phi, psi, f, pi, pj, Di, Dj, pol) 
            temp = np.trapezoid(integrand, theta, axis=2)
            integral = np.trapezoid(temp, phi, axis=1)  # → shape: (N_freq,)

            return np.real(integral)

        return gamma(pi, pj, Di, Dj, f, pol, psi)
    

    def overlap_PTA(f, pol, psi = 0):

        '''
        Compute the overlap reduction function for a set of pulsars (NANOGrav)

        Parameters:
        - f: array_like (Frequency in Hz)
        - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar breathing, 'l' for scalar longitudinal, 'I' for intensity, 'V' for circular)

        Return:
        - overlap: array_like (Overlap reduction function for a set of pulsars)
        '''

        N_pulsar, pulsar_xyz, D = detectors.get_NANOGrav_pulsars()
        
        overlap = np.zeros(len(f))


        for i in range(N_pulsar):
            for j in range(i +1, N_pulsar):
                overlap += Response.pairwise_overlap(f, pulsar_xyz[i], pulsar_xyz[j], D[i], D[j], pol, psi)
                
        return overlap
    
    # def pairwise_overlap_montecarlo(f, pi, pj, Di, Dj, pol, psi=0,  N_samples=10000):
    #     '''
    #     Compute the overlap reduction function between two pulsars using Monte Carlo integration.

    #     Parameters:
    #     - f: array_like (Frequency in Hz)
    #     - pi: array_like (Position of the first pulsar)
    #     - pj: array_like (Position of the second pulsar)
    #     - Di: float (Distance to the first pulsar in meters)
    #     - Dj: float (Distance to the second pulsar in meters)
    #     - pol: str (Polarization of the signal: 't', 'v', 's', 'l', 'I', or 'V')
    #     - psi: float (Polarization angle in radians, default is 0)

    #     Returns:
    #     - overlap: array_like (Overlap reduction function between two pulsars)
    #     '''

    #     import numpy as np

    #     c = 3e8  # Speed of light in m/s

    #     def gamma_integrand(theta, phi, psi, f, pi, pj, Di, Dj, pol):
    #         '''
    #         Integrand of the overlap reduction function for two pulsars
    #         '''
    #         Omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]

    #         Fp1 = AngularPatternFunction.F_pulsar(theta, phi, psi, pi)
    #         Fp2 = AngularPatternFunction.F_pulsar(theta, phi, psi, pj)

    #         f = f.reshape(len(f), 1, 1)

    #         exp1 = np.ones_like(len(f))(1-np.exp(-2j*np.pi*f*Di*(1+(np.einsum('iab,i->ab', Omega, pi)))/c))
    #         exp2 = np.ones_like(len(f))(1-np.exp(2j*np.pi*f*Dj*(1+(np.einsum('iab,i->ab', Omega, pj)))/c))

    #         if pol == 't':
    #             gamma_ij = 3 * (Fp1[0] * np.conj(Fp2[0]) + Fp1[1] * np.conj(Fp2[1])) * (1 / (8 * np.pi)) * exp1 * exp2
    #         elif pol == 'v':
    #             gamma_ij = 3 * (Fp1[2] * np.conj(Fp2[2]) + Fp1[3] * np.conj(Fp2[3])) * (1 / (8 * np.pi)) * exp1 * exp2
    #         elif pol == 's':
    #             gamma_ij = 3 * Fp1[4] * np.conj(Fp2[4]) * (1 / (8 * np.pi)) * exp1 * exp2
    #         elif pol == 'l':
    #             gamma_ij = 3 * Fp1[5] * np.conj(Fp2[5]) * (1 / (8 * np.pi)) * exp1 * exp2
    #         elif pol == 'I':
    #             gamma_ij = 3 * (Fp1[0] * np.conj(Fp2[0]) + Fp1[1] * np.conj(Fp2[1])) * (1 / (8 * np.pi)) * exp1 * exp2
    #         elif pol == 'V':
    #             gamma_ij = 3j * (Fp1[0] * np.conj(Fp2[1]) - Fp1[1] * np.conj(Fp2[0])) * (1 / (8 * np.pi)) * exp1 * exp2
    #         else:
    #             raise ValueError('Unknown polarization')

    #         return gamma_ij

    #     def gamma(pi, pj, Di, Dj, f, pol, psi, N_samples):
    #             # Create meshgrid (for consistency/debug), then randomly sample N points
    #             theta_grid = np.linspace(0, np.pi, 200)
    #             phi_grid = np.linspace(0, 2*np.pi, 200)
    #             Theta, Phi = np.meshgrid(theta_grid, phi_grid)


    #             # Sample N random indices
    #             indices = np.random.choice(len(Theta), size=N_samples)
    #             theta_samples = Theta[indices]
    #             phi_samples = Phi[indices]

    #             # Monte Carlo integration
    #             integrand = gamma_integrand(theta_samples, phi_samples, psi, f, pi, pj, Di, Dj, pol)
    #             mean_value = np.mean(integrand, axis=-1)  # average over samples
    #             integral = 4 * np.pi * mean_value
    #             return np.real(integral)

    #     return gamma(pi, pj, Di, Dj, f, pol, psi, N_samples)

    


    # def pairwise_overlap_adaptive(f, pi, pj, Di, Dj, pol, psi=0,):

    #     '''
    #     Compute the overlap reduction function between two pulsars using adaptive-like grid and vectorized frequencies.

    #     Parameters:
    #     - f: array_like (Frequency in Hz)
    #     - pi, pj: array_like (Position of the pulsars, 3D unit vectors)
    #     - Di, Dj: float (Distance to the pulsars in meters)
    #     - pol: str (Polarization: 't', 'v', 's', 'I', 'V')
    #     - psi: float (Polarization angle)
    #     - Ntheta, Nphi: int (Resolution of the angular grid)

    #     Returns:
    #     - overlap: array_like (ORF evaluated for each frequency)
    #     '''

    #     c = 299792458  # Speed of light

    #     Ntheta=100
    #     Nphi=100

    #     # Angular grid
    #     theta = np.linspace(0, np.pi, Ntheta)
    #     phi = np.linspace(0, 2*np.pi, Nphi)
    #     dtheta = theta[1] - theta[0]
    #     dphi = phi[1] - phi[0]

    #     Theta, Phi = np.meshgrid(theta, phi, indexing='ij')  # shape (Ntheta, Nphi)

    #     # Broadcast Theta and Phi to include a dimension for frequencies
    #     f = np.atleast_1d(f)
    #     F = f[:, None, None]  # shape (Nf, 1, 1)

    #     # Evaluate integrand (Theta, Phi are 2D; f is 1D; result will be (Nf, Ntheta, Nphi))
    #     def gamma_integrand(theta, phi, psi, f, pi, pj, Di, Dj, pol):
    #         Omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]  # shape (3, Ntheta, Nphi)
    #         dot_Omega_pi = np.einsum('ijk,i->jk', Omega, pi)
    #         dot_Omega_pj = np.einsum('ijk,i->jk', Omega, pj)

    #         exp1 = 1 - np.exp(-2j * np.pi * F * Di * (1 + dot_Omega_pi) / c)
    #         exp2 = 1 - np.exp(2j * np.pi * F * Dj * (1 + dot_Omega_pj) / c)

    #         Fp1 = AngularPatternFunction.F_pulsar(theta, phi, psi, pi)
    #         Fp2 = AngularPatternFunction.F_pulsar(theta, phi, psi, pj)

    #         # print('shape angular pattern function: ', Fp1[0].shape)
    #         # print('shape exp1: ', exp1.shape)

    #         sin_theta = np.sin(theta)
    #         if pol == 't':
    #             gamma = 3 * (Fp1[0]*np.conj(Fp2[0]) + Fp1[1]*np.conj(Fp2[1])) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 'v':
    #             gamma = 3 * (Fp1[2]*np.conj(Fp2[2]) + Fp1[3]*np.conj(Fp2[3])) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 's':
    #             gamma = 3 * Fp1[4]*np.conj(Fp2[4]) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 'l':
    #             gamma = 3 * Fp1[5]*np.conj(Fp2[5]) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 'I':
    #             gamma = 3 * (Fp1[0]*np.conj(Fp2[0]) + Fp1[1]*np.conj(Fp2[1])) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 'V':
    #             gamma = 3j * (Fp1[0]*np.conj(Fp2[1]) - Fp1[1]*np.conj(Fp2[0])) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         else:
    #             raise ValueError('Unknown polarization')

    #         return gamma  

    #     integrand = gamma_integrand(Theta, Phi, psi, f, pi, pj, Di, Dj, pol)

    #     # Double integration over theta and phi using np.trapezoid
    #     integral_theta = np.trapezoid(integrand, dx=dtheta, axis=1)     # shape (Nf, Nphi)
    #     integral = np.trapezoid(integral_theta, dx=dphi, axis=1)        # shape (Nf,)

    #     return np.real(integral)
    



    # def pairwise_overlap_adaptive_gradient(f, pi, pj, Di, Dj, pol, psi=0,
    #                             base_res=100, refine_factor=2, grad_threshold=0.05):
    #     '''
    #     Adaptive computation of the overlap reduction function between two pulsars.

    #     Parameters:
    #     - f: array_like, shape (Nf,)   – Frequencies (Hz)
    #     - pi, pj: array_like, shape (3,) – Positions of pulsars
    #     - Di, Dj: float – Distances to pulsars (m)
    #     - pol: str – Polarization
    #     - psi: float – Polarization angle
    #     - base_res: int – Base grid resolution for theta and phi
    #     - refine_factor: int – Refinement multiplier where gradient is high
    #     - grad_threshold: float – Threshold on gradient magnitude for refinement

    #     Returns:
    #     - overlap: array_like, shape (Nf,) – ORF evaluated at each frequency
    #     '''

    #     c = 299792458
    #     f = np.atleast_1d(f)
    #     Nf = len(f)
    #     overlap = np.zeros(Nf, dtype=np.complex128)

    #     # Base grid
    #     theta_base = np.linspace(0, np.pi, base_res)
    #     phi_base = np.linspace(0, 2*np.pi, base_res)
    #     dtheta = theta_base[1] - theta_base[0]
    #     dphi = phi_base[1] - phi_base[0]
    #     Theta_base, Phi_base = np.meshgrid(theta_base, phi_base, indexing='ij')

    #     def gamma_integrand(theta, phi, freq, pi, pj, Di, Dj, pol, psi):
    #         Omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]  # (3, Ntheta, Nphi)
    #         dot_Omega_pi = np.einsum('ijk,i->jk', Omega, pi)
    #         dot_Omega_pj = np.einsum('ijk,i->jk', Omega, pj)
    #         exp1 = 1 - np.exp(-2j * np.pi * freq * Di * (1 + dot_Omega_pi) / c)
    #         exp2 = 1 - np.exp( 2j * np.pi * freq * Dj * (1 + dot_Omega_pj) / c)
    #         Fp1 = AngularPatternFunction.F_pulsar(theta, phi, psi, pi)
    #         Fp2 = AngularPatternFunction.F_pulsar(theta, phi, psi, pj)
    #         sin_theta = np.sin(theta)

    #         if pol == 't':
    #             integrand = 3 * (Fp1[0]*np.conj(Fp2[0]) + Fp1[1]*np.conj(Fp2[1])) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 'v':
    #             integrand = 3 * (Fp1[2]*np.conj(Fp2[2]) + Fp1[3]*np.conj(Fp2[3])) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 's':
    #             integrand = 3 * Fp1[4]*np.conj(Fp2[4]) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 'l':
    #             integrand = 3 * Fp1[5]*np.conj(Fp2[5]) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 'I':
    #             integrand = 3 * (Fp1[0]*np.conj(Fp2[0]) + Fp1[1]*np.conj(Fp2[1])) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         elif pol == 'V':
    #             integrand = 3j * (Fp1[0]*np.conj(Fp2[1]) - Fp1[1]*np.conj(Fp2[0])) * (1/(8*np.pi)) * sin_theta * exp1 * exp2
    #         else:
    #             raise ValueError('Unknown polarization')

    #         return integrand

    #     for i in range(Nf):
    #         freq = f[i]

    #         # Step 1: Compute integrand on base grid
    #         integrand_base = gamma_integrand(Theta_base, Phi_base, freq, pi, pj, Di, Dj, pol, psi)

    #         # Step 2: Compute gradient magnitude
    #         grad_theta, grad_phi = np.gradient(np.abs(integrand_base), dtheta, dphi)
    #         grad_mag = np.sqrt(grad_theta**2 + grad_phi**2)
    #         grad_max = np.max(grad_mag)
    #         mask_refine = grad_mag > grad_threshold * grad_max

    #         # Step 3: Define refined grid
    #         refined_theta = theta_base.copy()
    #         refined_phi = phi_base.copy()

    #         if np.any(mask_refine):
    #             idx_theta, idx_phi = np.where(mask_refine)
    #             refined_points = set()
    #             for it, ip in zip(idx_theta, idx_phi):
    #                 for dt in range(refine_factor):
    #                     for dp in range(refine_factor):
    #                         t = theta_base[it] + (dt / refine_factor) * dtheta
    #                         p = phi_base[ip] + (dp / refine_factor) * dphi
    #                         if 0 <= t <= np.pi and 0 <= p <= 2*np.pi:
    #                             refined_points.add((t, p))
    #             refined_points = np.array(list(refined_points))
    #             refined_theta = np.unique(np.concatenate([refined_theta, refined_points[:, 0]]))
    #             refined_phi = np.unique(np.concatenate([refined_phi, refined_points[:, 1]]))

    #         Theta_ref, Phi_ref = np.meshgrid(refined_theta, refined_phi, indexing='ij')

    #         # Step 4: Interpolate integrand on refined grid
    #         interp_re = RectBivariateSpline(theta_base, phi_base, np.real(integrand_base))
    #         interp_im = RectBivariateSpline(theta_base, phi_base, np.imag(integrand_base))
    #         integrand_ref = interp_re(refined_theta, refined_phi) + 1j * interp_im(refined_theta, refined_phi)

    #         # Step 5: Integrate on refined grid
    #         integral_theta = np.trapezoid(integrand_ref, dx=np.diff(refined_theta).mean(), axis=0)
    #         integral = np.trapezoid(integral_theta, dx=np.diff(refined_phi).mean())
    #         overlap[i] = integral

    #     return np.real(overlap)
    