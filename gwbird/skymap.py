import numpy as np
from numpy import pi, cos, sin, sqrt
import matplotlib.pyplot as plt
from gwbird import detectors 
from gwbird.utils import c
import healpy as hp
from healpy.newvisufunc import projview, newprojplot


plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.family'] = 'serif'


'''
The skymap module contains the following classes:

    - Basis: contains the orthonormal basis in the direction of the incoming GW signal
    - PolarizationTensors: contains the polarization modes in the general orthonormal basis (m,n, Omega)
    - TransferFunction: contains the transfer function to take into account the large antenna limit
    - AngularPatternFunction: contains the angular pattern function: detector response to an incoming GW signal
    - Skymaps: contains the AntennaPattern function to compute the overlap map
'''


class Basis:

    def u_v_Omega_basis(theta, phi):

        '''
        Orthonormal basis in the direction of the incoming GW signal

        Parameters:
        - theta: float/array_like (polar angle in [0, pi])
        - phi: float/array_like (azimuthal angle in [0, 2pi])

        Returns:
        - u, v, Omega: array_like (orthonormal basis)
        '''
        
        u = np.stack([
            np.cos(phi) * np.cos(theta),
            np.sin(phi) * np.cos(theta),
            -np.sin(theta)
        ], axis=0) 
        
        v = np.stack([
            np.sin(phi), 
            -np.cos(phi), 
            np.zeros_like(theta)
        ], axis=0)  


        Omega = np.stack([
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta)
        ], axis=0)  

        return u, v, Omega
    
    def m_n_Omega_basis(theta, phi, psi):

        '''
        Orthonormal basis in the general direction of the incoming GW signal with the rotation angle psi

        Parameters:
        - theta: float/array_like (polar angle in [0, pi])
        - phi: float/array_like (azimuthal angle in [0, 2pi])
        - psi: float/array_like (polarization angle in [0, pi])

        Returns:
        - m, n, Omega: array_like (orthonormal basis)
        '''

        u, v, Omega = Basis.u_v_Omega_basis(theta, phi)

        m = np.cos(psi) * u - np.sin(psi) * v
        n = np.sin(psi) * u + np.cos(psi) * v
        Omega = Omega

        return m, n, Omega

class PolarizationTensors:

    def e_pol(theta, phi, psi):  

        '''
        Polarization modes in the general orthonormal basis (m,n, Omega)

        Parameters:
        - theta: float/array_like (polar angle in [0, pi])
        - phi: float/array_like (azimuthal angle in [0, 2pi])
        - psi: float/array_like (polarization angle in [0, pi])

        Returns:
        - e_plus, e_cross, e_x, e_y, e_b, e_l: array_like (polarization tensors)
        '''

        m, n, Omega = Basis.m_n_Omega_basis(theta, phi, psi)
        e_plus = np.einsum('i...,k...',m,m)-np.einsum('i...,k...',n,n)
        e_cross = np.einsum('i...,k...',m,n)+np.einsum('i...,k...',n,m)
        e_x = np.einsum('i...,k...',m,Omega)+np.einsum('i...,k...',Omega,m)
        e_y = np.einsum('i...,k...',n,Omega)+np.einsum('i...,k...',Omega,n)
        e_b = np.einsum('i...,k...',m,m)+np.einsum('i...,k...',n,n)
        e_l = np.einsum('i...,k...',Omega,Omega) * sqrt(2)

        return e_plus, e_cross, e_x, e_y, e_b, e_l
    


class TransferFunction:

    def transfer_function(L, l, f, theta, phi, psi):

        '''
        Transfer function to take into account the large antenna limit
    
        Parameters:
        - L: float (arm length)
        - l: array_like (unit vector of the detector arm)
        - f: array_like (frequency)
        - theta: float/array_like (polar angle in [0, pi])
        - phi: float/array_like (azimuthal angle in [0, 2pi])
        - psi: float/array_like (polarization angle in [0, pi])

        Returns:
        - transfer function: (array_like)
        '''

        Omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]
        f = f.reshape((len(f), 1, 1))

        inner = np.einsum('iab,i', Omega, l)
        f_star = c / 2 / pi / L
        return 1 / 2 * (np.sinc(f / 2 / pi / f_star * (1 - inner)) * np.exp(-1j * f / 2 / f_star * (3 + inner)) \
                        + np.sinc(f / 2 / pi / f_star * (1 + inner)) * np.exp(-1j * f / 2 / f_star * (1 + inner)))


# detector tensor + angular pattern function

class AngularPatternFunction:

    def F(theta, phi, psi, ce, e1, e2, f, L): 

        '''
        Angular pattern function: detector response to an incoming GW signal

        Parameters:
        - theta: float/array_like (polar angle in [0, pi])
        - phi: float/array_like (azimuthal angle in [0, 2pi])
        - psi: float/array_like (polarization angle in [0, pi])
        - ce: array_like (vector pointing towards the detector location)
        - e1: array_like (unit vector of the detector arm 1)
        - e2: array_like (unit vector of the detector arm 2)
        - f: array_like (frequency)
        - L: float (arm length)

        Returns:
        - F_plus, F_cross, F_x, F_y, F_b, F_l: array_like (angular pattern function)
        '''

        e_plus, e_cross, e_x, e_y, e_b, e_l = PolarizationTensors.e_pol(theta, phi, psi)
        omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]
        f = f.reshape(len(f), 1, 1)
        f_star = c/2/pi
        tr_arm1 = TransferFunction.transfer_function(L, e1, f, theta, phi, psi)
        tr_arm2 = TransferFunction.transfer_function(L, e2, f, theta, phi, psi)
        exp_c = np.exp(-1j*f/f_star * (np.einsum('iab,i->ab', omega, ce))) 
        
        F_plus = 0.5 * exp_c * ( tr_arm1 * np.einsum('i,j,lmij', e1, e1, e_plus) - tr_arm2 * np.einsum('i,j,lmij', e2, e2, e_plus) )
        F_cross = 0.5 * exp_c * ( tr_arm1 * np.einsum('i,j,lmij', e1, e1, e_cross) - tr_arm2 * np.einsum('i,j,lmij', e2, e2, e_cross) )
        F_x = 0.5 * exp_c * ( tr_arm1 * np.einsum('i,j,lmij', e1, e1, e_x) - tr_arm2 * np.einsum('i,j,lmij', e2, e2, e_x) )
        F_y = 0.5 * exp_c * ( tr_arm1 * np.einsum('i,j,lmij', e1, e1, e_y) - tr_arm2 * np.einsum('i,j,lmij', e2, e2, e_y) )
        F_b = 0.5 * exp_c * ( tr_arm1 * np.einsum('i,j,lmij', e1, e1, e_b) - tr_arm2 * np.einsum('i,j,lmij', e2, e2, e_b) )
        F_l = 0.5 * exp_c * ( tr_arm1 * np.einsum('i,j,lmij', e1, e1, e_l) - tr_arm2 * np.einsum('i,j,lmij', e2, e2, e_l) )

        return F_plus, F_cross, F_x, F_y, F_b, F_l 
    
    def F_pulsar(theta, phi, psi, p): # https://arxiv.org/pdf/1306.5394, https://arxiv.org/abs/2407.14460
        '''
        Compute the antenna pattern functions for a given direction of the source and polarization angle psi

        Parameters:
        - theta: float/array_like (polar angle in [0, pi])
        - phi: float/array_like (azimuthal angle in [0, 2pi])
        - psi: float/array_like (polarization angle in [0, pi])
        - p: array_like (unit vector pointing towards the pulsar)

        Returns:
        - F_plus, F_cross, F_x, F_y, F_b, F_l: array_like (angular pattern function)
        '''
        Omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]
        p = p.reshape(3, 1, 1)
        D = 0.5 * np.einsum('i..., j... -> ij...', p, p)  / (1 + np.einsum('ijk, ijk -> jk', Omega, p))
        e_plus, e_cross, e_x, e_y, e_b, e_l  = PolarizationTensors.e_pol(theta, phi, psi)
        F_plus = np.einsum('ijkl, klij ->kl ', D, e_plus)
        F_cross = np.einsum('ijkl, klij ->kl', D, e_cross) 
        F_x = np.einsum('ijkl, klij ->kl', D, e_x)
        F_y = np.einsum('ijkl, klij ->kl', D, e_y)
        F_b = np.einsum('ijkl, klij ->kl', D, e_b)
        F_l = np.einsum('ijkl, klij ->kl', D, e_l) 
        return F_plus, F_cross, F_x, F_y, F_b, F_l


class Skymaps:

    def AntennaPattern(det1, det2, f, psi, pol, nside=32, shift_angle=None):
            
        '''
        Antenna pattern function: detector response to an incoming GW signal

        Parameters:
        - det1, det2: str or list of str
            The name of the detector(s) to consider.
            The names must be in the list of detectors available.
            The list of available detectors can be obtained by calling the function detectors.available_detectors().
            The names of the detectors are case sensitive.
            If you want to provide a custom detector, you can provide the following information in a list:

            H = [c, xA, xB, l, name]

            - c: array_like of length 3 (Position of the detector in the Earth-centered frame in meters)
            - xA: array_like of length 3 (Unit vector pointing towards the detector in the Earth-centered frame)
            - xB: array_like of length 3 (Unit vector pointing towards the detector in the Earth-centered frame)
            - l: float (Length of the arm in meters)
            - name: str (Name of the detector)
        - f: array_like (frequency)
        - psi: float/array_like (polarization angle in [0, pi])
        - pol: str (polarization type: 't', 'v', 's, 'I', 'V')
        - nside: int (Healpix resolution parameter)
        - shift_angle: float (angle to rotate the detector, only for ET 2L)

        Returns:
        - selected_map: array_like (Antenna pattern Function map)

        '''
        

        def get_detector(d):
            if isinstance(d, str):
                return detectors.detector(d, shift_angle)  # Usa il modulo corretto
            elif isinstance(d, list) and len(d) == 5:
                # Se è un detector customizzato
                c, xA, xB, l, name = d
                return np.array(c), np.array(xA), np.array(xB), l, name
            else:
                raise ValueError(f"Invalid detector format: {d}")

        
        ec1, u1, v1, l1, which_det1 = get_detector(det1)
        ec2, u2, v2, l2, which_det2 = get_detector(det2)
    
        npix = hp.nside2npix(nside)
        theta,phi = hp.pix2ang(nside, np.arange(npix))

        theta = theta.reshape(-1, 1)
        phi = phi.reshape(-1, 1)
        
        f = np.array([f]) 
        f = f.reshape(len(f), 1, 1)
                
        F1 = AngularPatternFunction.F(theta, phi, psi, ec1, u1, v1, f, l1)
        F2 = AngularPatternFunction.F(theta, phi, psi, ec2, u2, v2, f, l2)

        f_star = c/2/pi
        omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]
        exp_c1 = np.exp(1j*f/f_star * (np.einsum('iab,i->ab', omega, ec1))) 
        exp_c2 = np.exp(1j*f/f_star * (np.einsum('iab,i->ab', omega, ec2)))

        F1 = np.array(F1)
        F2 = np.array(F2)

        overlap_map = {
                    "t": 5/(8*pi)*np.real(F1[0]*exp_c1*np.conj(F2[0]*exp_c2) + F1[1]*exp_c1*np.conj(F2[1]*exp_c2)) ,
                    "v": 5/(8*pi)*np.real(F1[2]*exp_c1*np.conj(F2[2]*exp_c2) + F1[3]*exp_c1*np.conj(F2[3]*exp_c2)) ,
                    "s": 15/(4*pi)*np.real(F1[4]*exp_c1*np.conj(F2[4]*exp_c2)),
                    "I": 5/(8*pi)*np.real(F1[0]*exp_c1*np.conj(F2[0]*exp_c2) + F1[1]*exp_c1*np.conj(F2[1]*exp_c2)),
                    "V": 1j*(5/(8*pi))*np.real(F1[0]*exp_c1*np.conj(F2[1]*exp_c2) - F1[1]*exp_c1*np.conj(F2[0]*exp_c2))
                    }

        if pol not in overlap_map:
            raise ValueError("Invalid polarization type. Choose from 't', 'v', 's', 'I', 'V'")
    
        selected_map = np.squeeze(np.real(overlap_map[pol]))

        plt.close('all')  
        plt.rcParams.update({'font.size': 15})

        hp.mollview(selected_map, flip="astro", coord=["C"], title=pol.capitalize(), cmap='viridis')
        hp.visufunc.graticule(dpar=45, dmer=60, coord='C', local=True, color='white')

        plt.show()

        return  selected_map
        
        
        
        
