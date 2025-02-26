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

#obs = det.Observatories()


# descrizione del codice


class Basis:


    def u_v_Omega_basis(theta, phi):

        '''
        Orthonormal basis in the direction of the incoming GW signal

        Parameters:
        theta: polar angle in [0, pi]
        phi: azimuthal angle in [0, 2pi]

        Returns:
        u, v, Omega: orthonormal basis

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
        General basis polarization angle psi

        Parameters:
        theta: polar angle in [0, pi]
        phi: azimuthal angle in [0, 2pi]
        psi: polarization angle in [0, 2pi]

        Returns:
        m, n, Omega: orthonormal basis with the rotation angle psi
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
        theta: polar angle in [0, pi]
        phi: azimuthal angle in [0, 2pi]
        psi: polarization angle in [0, 2pi]

        Returns:
        e_plus, e_cross, e_x, e_y, e_b, e_l: polarization tensors
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
        L: arm length
        l: unit vector
        f: frequency
        theta: polar angle in [0, pi]
        phi: azimuthal angle in [0, 2pi]
        psi: polarization angle in [0, 2pi]

        Returns:
        transfer function
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
        theta: polar angle in [0, pi]
        phi: azimuthal angle in [0, 2pi]
        psi: polarization angle in [0, 2pi]
        ce: vector of the centre of the Detector
        e1: unit vector of the Detector 1st arm
        e2: unit vector of the Detector 2nd arm
        f: frequency
        L: arm length
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


class Skymaps:

    def AntennaPattern(det1, det2, f, psi, pol, nside=32, shift_angle=None):
            
            '''

            Antenna pattern function: detector response to an incoming GW signal

            Parameters:
            det1, det2: str or list of str
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
            f: frequency array
            psi: polarization angle in [0, pi]

            Returns:
            overlap_map: overlap map, array of shape (npix,)

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
            
            f = np.array([f]) # check if this is correct
            f = f.reshape(len(f), 1, 1)
                 
            F1 = AngularPatternFunction.F(theta, phi, psi, ec1, u1, v1, f, l1)
            F2 = AngularPatternFunction.F(theta, phi, psi, ec2, u2, v2, f, l2)
            
            overlap_map = {
            "t": 5/(8*pi)*np.real(F1[0]*np.conj(F2[0]) + F1[1]*np.conj(F2[1])) ,
            "v": 5/(8*pi)*np.real(F1[2]*np.conj(F2[2]) + F1[3]*np.conj(F2[3])) ,
            "s": 15/(4*pi)*np.real(F1[4]*np.conj(F2[4])),
            "I": 5/(8*pi)*np.real(F1[0]*np.conj(F2[0]) + F1[1]*np.conj(F2[1])),
            "V": 1j*(5/(8*pi))*np.real(F1[0]*np.conj(F2[1]) - F1[1]*np.conj(F2[0]))
            }

            if pol not in overlap_map:
                raise ValueError("Invalid polarization type. Choose from 'tensor', 'vector', or 'scalar'.")
            

            selected_map = np.squeeze(overlap_map[pol])

            plt.close('all')  # Chiude tutte le figure aperte per evitare il doppio plot
            plt.rcParams.update({'font.size': 15})

            hp.mollview(selected_map, flip="astro", coord=["C"], title=pol.capitalize(), cmap='viridis')
            hp.visufunc.graticule(dpar=45, dmer=60, coord='C', local=True, color='white')

            plt.show()

  
            return  selected_map
            
            
    
            

    # def AntennaPattern(det1, det2, nside, f, psi, shift_angle=None):
            
    #         # specificare la polarizzazione
        
    #         ec1, u1, v1, l1, which_det1 = det.detector(det1, shift_angle)
    #         ec2, u2, v2, l2, which_det2 = det.detector(det2, shift_angle)
        
    #         npix = hp.nside2npix(nside)
    #         theta,phi = hp.pix2ang(nside, np.arange(npix))

    #         theta = theta.reshape(-1, 1)
    #         phi = phi.reshape(-1, 1)
            
    #         f = np.array([f]) # check if this is correct
    #         f = f.reshape(len(f), 1, 1)
                 
    #         F1 = AngularPatternFunction.F(theta, phi, psi, ec1, u1, v1, f, l1)
    #         F2 = AngularPatternFunction.F(theta, phi, psi, ec2, u2, v2, f, l2)

    #         if det1 == 'LISA 1' or det1 == 'LISA 2' or det1 == 'LISA 3': # in det starts for LISA...
    #             overlap_t = 1/(4*np.pi) * np.real(F1[0]*np.conj(F2[0]) + F1[1]*np.conj(F2[1]))
    #             overlap_v = 1/(4*np.pi) * np.real(F1[2]*np.conj(F2[2]) + F1[3]*np.conj(F2[3]))
    #             overlap_s = 3/(2*np.pi) * np.real(F1[4]*np.conj(F2[4]))

    #         else:
    #             overlap_t = 5/(8*pi)*np.real(F1[0]*np.conj(F2[0]) + F1[1]*np.conj(F2[1])) 
    #             overlap_v = 5/(8*pi)*np.real(F1[2]*np.conj(F2[2]) + F1[3]*np.conj(F2[3])) 
    #             overlap_s = 15/(4*pi)*np.real(F1[4]*np.conj(F2[4]))
                

    #         maps = [overlap_t,
    #                 overlap_v,
    #                 overlap_s
    #                 ]
            
    #         maps_flat = np.zeros((len(maps), npix))

    #         for i in range(len(maps)):
    #             maps_flat[i]= np.squeeze(maps[i])

    #         maps = maps_flat
            
    #         fig, ax = plt.subplots(1,3,figsize=(20, 4))
    #         ax = ax.flatten()
    #         plt.rcParams.update({'font.size': 17})
    #         titles = ["tensor", "vector", "scalar"]          
    #         titles = np.array(titles).flatten()
    #         plt.suptitle(r"$\gamma(\hat{{\Omega}})$ ({0} - {1})".format(which_det1, which_det2))
            
    #         for i in range(len(maps)):
    #                 plt.axes(ax[i])
    #                 ax[i].set_title(r"$\Gamma$({0} - {1})".format(which_det1, which_det2))
    #                 hp.mollview(maps[i], 
    #                         flip="astro",
    #                         coord=["C"],
    #                         title=titles[i],
    #                         cmap='viridis',
    #                         min = '{:.2f}'.format(np.min(maps[i])),
    #                         max = '{:.2f}'.format(np.max(maps[i])),
    #                         notext=True,
    #                         hold=True)    
    #                 hp.visufunc.graticule(dpar=45, dmer=60, coord='C', local=True, color='white' )       
            
            
    #         fig = plt.gcf()
    #         axs = fig.get_axes()
            
    #         cbar_ax = axs[-1]
            
    #         if hasattr(cbar_ax, "set_yticklabels"):
                
    #             ticks = cbar_ax.get_yticks()
    #             new_tick_labels = ["{:.1g}".format(tick) for tick in ticks]
    #             cbar_ax.set_yticklabels(new_tick_labels)
            
    #         plt.show()
            
    #         return  maps
            
            
    