import numpy as np
from numpy import pi, cos, sin, sqrt
import matplotlib.pyplot as plt
from nest import detectors as det
from nest import overlap as overlap 

from tqdm import tqdm
from numba import  jit
import matplotlib.pyplot as plt
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

obs = det.Observatories()

c = 299792458 # speed of light


# def u_v_Omega_basis(theta, phi):
#     u = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*sin(phi), -np.sin(theta)])
#     v = np.array([-np.sin(phi), np.cos(phi), 0])
#     Omega = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
#     return u, v, Omega


# def m_n_Omega_basis(theta, phi, psi):
#     u, v, Omega = u_v_Omega_basis(theta, phi)
#     m = u*np.cos(psi) + v*np.sin(psi)
#     n = -u*np.sin(psi) + v*np.cos(psi)
#     return m, n, Omega


# def e_pol(theta, phi, psi): #polarization tensors    
#     m, n, Omega = m_n_Omega_basis(theta, phi, psi)
#     e_plus = np.einsum('i,j->ij', m, m)- np.einsum('i,j->ij', n, n)
#     e_cross = np.einsum('i,j->ij',m, n) + np.einsum('i,j->ij',n, m)
#     e_x = np.einsum('i,j->ij',m, Omega) + np.einsum('i,j->ij',Omega, m)
#     e_y = np.einsum('i,j->ij',n, Omega) + np.einsum('i,j->ij',Omega, n)
#     e_b = np.einsum('i,j->ij',m,m) + np.einsum('i,j->ij',n,n)
#     e_l = sqrt(2)*(np.einsum('i,j->ij',Omega, Omega))
#     return e_plus, e_cross, e_x, e_y, e_b, e_l

# # transfer function

# def transfer_function(L, l, f, theta, phi, psi):
#     # Assicurarsi che f sia un array 2D con la stessa forma di theta e phi
#     # Calcola la funzione di trasferimento
#     omega = - m_n_Omega_basis(theta, phi, psi)[2]
    
#     exp3 = np.exp(-1j*pi*f*L/c*(3+np.einsum('i,i->', omega, l)))
#     exp1 = np.exp(-1j*pi*f*L/c*(1+np.einsum('i,i->', omega, l)))
#     sinc3 = np.sinc(f*L/c*(1-np.einsum('i,i->', omega, l)))
#     sinc1 = np.sinc(f*L/c*(1+np.einsum('i,i->', omega, l)))
#     return 0.5*(exp1*sinc1 + exp3*sinc3)


# # detector tensor

# def D_tensor(e1, e2, L, f, theta, phi, psi): #detector tensor
#     #L = 2.5e9
#     tf1 = transfer_function(L, e1, f, theta, phi, psi)
#     tf2 = transfer_function(L, e2, f, theta, phi, psi)
#     out1 = np.einsum('i,j-> ij',e1, e1)
#     out2 = np.einsum('i,j-> ij',e2, e2)
#     p1 = np.einsum('ij,->ij', out1, tf1)
#     p2 = np.einsum('ij,->ij', out2, tf2)
#     D = 0.5*( p1 - p2)
#     return D

# # angular patter function

# def F(theta, phi, psi, f, e1, e2, L): #angular pattern function
#     e_plus, e_cross, e_x, e_y, e_b, e_l = e_pol(theta, phi, psi)
#     D = D_tensor(e1, e2, L, f, theta, phi, psi)  #detector tensor with transfer function
#     F_plus = np.einsum('ij,ij->', D, e_plus)
#     F_cross = np.einsum('ij,ij->', D, e_cross)
#     F_x = np.einsum('ij,ij->', D, e_x)
#     F_y = np.einsum('ij,ij->', D, e_y)
#     F_b = np.einsum('ij,ij->', D, e_b)
#     F_l = np.einsum('ij,ij->', D, e_l)
#     return np.real(F_plus), np.real(F_cross), np.real(F_x), np.real(F_y), np.real(F_b), np.real(F_l) 


class Basis:
    @staticmethod
    def u_v_Omega_basis(theta, phi):

        '''
        Coordinate system rotated by angles theta and phi 
        wrt the orthonormal coordinate system x,y,z
        '''

        u0 = np.cos(theta) * np.cos(phi)
        u1 = np.cos(theta) * np.sin(phi)
        u2 = -np.sin(theta)

        v0 = -np.sin(phi)
        v1 = np.cos(phi)
        v2 = np.zeros_like(phi)  

        Omega0 = np.sin(theta) * np.cos(phi)
        Omega1 = np.sin(theta) * np.sin(phi)
        Omega2 = np.cos(theta)

        u = np.stack((u0, u1, u2), axis=0)
        v = np.stack((v0, v1, v2), axis=0)
        Omega = np.stack((Omega0, Omega1, Omega2), axis=0)

        return u, v, Omega


    @staticmethod
    def m_n_Omega_basis(theta, phi, psi):

        '''
        General polarization angle psi
        '''

        u, v, Omega = Basis.u_v_Omega_basis(theta, phi)
        m = u * np.cos(psi) + v * np.sin(psi)
        n = -u * np.sin(psi) + v * np.cos(psi)
        return m, n, Omega

class PolarizationTensors:
    @staticmethod
    def e_pol(theta, phi, psi):  

        '''
        Polarization modes in the general orthonormal basis (m,n, Omega)
        '''

        m, n, Omega = Basis.m_n_Omega_basis(theta, phi, psi)
        e_plus = np.einsum('iab,jab->ijab', m, m) - np.einsum('iab,jab->ijab', n, n)
        e_cross = np.einsum('iab,jab->ijab', m, n) + np.einsum('iab,jab->ijab', n, m)
        e_x = np.einsum('iab,jab->ijab', m, Omega) + np.einsum('iab,jab->ijab', Omega, m)
        e_y = np.einsum('iab,jab->ijab', n, Omega) + np.einsum('iab,jab->ijab', Omega, n)
        e_b = np.einsum('iab,jab->ijab', m, m) + np.einsum('iab,jab->ijab', n, n)
        e_l = sqrt(2) * (np.einsum('iab,jab->ijab', Omega, Omega))
        return e_plus, e_cross, e_x, e_y, e_b, e_l

class TransferFunction:
    @staticmethod
    def transfer_function(L, l, f, theta, phi, psi):

        '''
        Transfer function to take into account the large antenna limit
        '''
        # fix 

        omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]
        f = f.reshape((len(f), 1, 1))
        inner = np.einsum('iab,i->ab', omega, l)
        f_star = c / 2 / np.pi / L
        return 1 / 2 * (np.sinc(f / 2 / np.pi / f_star * (1 - inner)) * np.exp(-1j * f / 2 / f_star * (3 + inner)) \
                        + np.sinc(f / 2 / np.pi / f_star * (1 + inner)) * np.exp(-1j * f / 2 / f_star * (1 + inner)))


# detector tensor + angular pattern function

class AngularPatternFunction:
    @staticmethod
    def F(theta, phi, psi, ce, e1, e2, f, L): 

        '''
        Angular pattern function: detector response to an incoming GW signal
        '''

        e_plus, e_cross, e_x, e_y, e_b, e_l = PolarizationTensors.e_pol(theta, phi, psi)
        omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]
        f = f.reshape(len(f), 1, 1)
        f_star = c/2/np.pi
        tr_arm1 = TransferFunction.transfer_function(L, e1, f, theta, phi, psi)
        tr_arm2 = TransferFunction.transfer_function(L, e2, f, theta, phi, psi)
        exp_c = np.exp(-1j*f/f_star * (np.einsum('iab,i->ab', omega, ce))) 
        tr_arm1 = tr_arm1*exp_c 
        tr_arm2 = tr_arm2*exp_c
        D1 = np.einsum('i,j-> ij',e1, e1)
        D2 = np.einsum('i,j-> ij',e2, e2)
        p1 = np.einsum('ij, xab->ijxab', D1, tr_arm1)
        p2 = np.einsum('ij, xab->ijxab', D2, tr_arm2)
        D = 0.5*( p1 - p2)
        F_plus = np.einsum('ijxab,ijab->xab', D, e_plus)
        F_cross = np.einsum('ijxab,ijab->xab', D, e_cross)  
        F_x = np.einsum('ijxab,ijab->xab', D, e_x)
        F_y = np.einsum('ijxab,ijab->xab', D, e_y)
        F_b = np.einsum('ijxab,ijab->xab', D, e_b)
        F_l = np.einsum('ijxab,ijab->xab', D, e_l)

        return F_plus, F_cross, F_x, F_y, F_b, F_l
    




class Skymaps:
            
        
    @staticmethod
    def AntennaPattern(det1, det2, nside, f, psi, shift_angle=None):
        
            observatories = det.Observatories()
            ec1, u1, v1, l1, which_det1 = det.detector(det1, shift_angle)
            ec2, u2, v2, l2, which_det2 = det.detector(det2, shift_angle)
        
            npix = hp.nside2npix(nside)
            theta,phi = hp.pix2ang(nside, np.arange(npix))

            theta = theta.reshape(-1, 1)
            phi = phi.reshape(-1, 1)
            
            f = np.array([f])
            f = f.reshape(len(f), 1, 1)

            print(f.shape)
            
                 
            F1 = AngularPatternFunction.F(theta, phi, psi, ec1, u1, v1, f, l1)
            F2 =  AngularPatternFunction.F(theta, phi, psi, ec2, u2, v2, f, l2)
            overlap_t= 5/(8*np.pi)*(F1[0]*F2[0] + F1[1]*F2[1]) 
            overlap_v = 5/(8*np.pi)*(F1[2]*F2[2] + F1[3]*F2[3]) 
            overlap_s = 3/(2*np.pi)*(F1[4]*F2[4])
                

            maps = [overlap_t,
                    overlap_v,
                    overlap_s
                    ]
            
            maps_flat = np.zeros((len(maps), npix))

            for i in range(len(maps)):
                # Estrai la mappa e rimuovi dimensioni extra
                maps_flat[i]= np.squeeze(maps[i])

            maps = maps_flat
            
            fig, ax = plt.subplots(1,3,figsize=(20, 4))
            ax = ax.flatten()
            plt.rcParams.update({'font.size': 17})
            titles = ["tensor", "vector", "scalar"]          
            titles = np.array(titles).flatten()
            plt.suptitle(r"$\gamma(\hat{{\Omega}})$ ({0} - {1})".format(which_det1, which_det2))
            
            for i in range(len(maps)):
                    plt.axes(ax[i])
                    ax[i].set_title("$\Gamma$({0} - {1})".format(which_det1, which_det2))
                    hp.mollview(maps[i], 
                            flip="astro",
                            coord=["C"],
                            title=titles[i],
                            cmap='viridis',
                            min = '{:.2f}'.format(np.min(maps[i])),
                            max = '{:.2f}'.format(np.max(maps[i])),
                            notext=True,
                            hold=True)    
                    hp.visufunc.graticule(dpar=45, dmer=60, coord='C', local=True, color='white' )       
            
            
                # Access the current figure and its axes to find the colorbar
            fig = plt.gcf()
            axs = fig.get_axes()
            
            # The last axes instance should be the colorbar in a typical hp.mollview layout
            cbar_ax = axs[-1]
            
            if hasattr(cbar_ax, "set_yticklabels"):
                # Get current ticks
                ticks = cbar_ax.get_yticks()
                
                # Format tick labels to one significant figure
                # You might need to adjust the formatting based on the range of your data
                new_tick_labels = ["{:.1g}".format(tick) for tick in ticks]
                
                # Set new tick labels
                cbar_ax.set_yticklabels(new_tick_labels)
            
            plt.show()
            
            return  maps
            
            
    