import jax
from jax import jit
import jax.numpy as np
import numpy
from jax.numpy import cos, sin, pi, sqrt
from nest import detectors as det
import matplotlib.pyplot as plt
import mpmath as mp


plt.rcParams['figure.dpi'] = 200
REarth = 6.371 * 1e6 #m
c = 3*1e8


# basis

class Basis:
    @staticmethod
    @jit
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
    @jit
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
    def transfer_function(L, e, f, theta, phi, psi):
        omega = Basis.m_n_Omega_basis(theta, phi, psi)[2]
        l = e
        f = f.reshape((len(f), 1, 1))
        inner = np.einsum('iab,i->ab', omega, l)
        f_star = c / 2 / np.pi / L
        return 1 / 2 * (np.sinc(f / 2 / np.pi / f_star * (1 - inner)) * np.exp(-1j * f / 2 / f_star * (3 + inner)) \
                        + np.sinc(f / 2 / np.pi / f_star * (1 + inner)) * np.exp(-1j * f / 2 / f_star * (1 + inner)))


# detector tensor + angular pattern function

class AngularPatternFunction:
    @staticmethod
    def F(theta, phi, psi, ce, e1, e2, f, L): #angular pattern function
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
    

# # angular overlap redunction function - angular response

# #      (overlap reduction function averaged over the sky)

class Response:
    @staticmethod
    def integrand(x, y, psi, c1, xA1, xB1, c2, xA2, xB2, c, f, L, pol):

        F1 = AngularPatternFunction.F(x, y, psi, c1, xA1, xB1, f, L)
        F2 = AngularPatternFunction.F(x, y, psi, c2, xA2, xB2, f, L)

        f = f.reshape(len(f), 1, 1)

        if (pol == 't'):
            return (5/(8*pi))*\
                ( F1[0]* np.conj( F2[0]) \
                + F1[1] *np.conj(F2[1])) \
                *sin(x)
        
        if (pol == 'v'):
            return (5/(8*pi))*\
                ( F1[2]* np.conj( F2[2]) \
                + F1[3] *np.conj(F2[3])) \
                *sin(x)
        
        if (pol == 's'):
            return (15/(4*pi))*\
                ( F1[4]* np.conj( F2[4]) ) \
                * sin(x)

    @staticmethod
    def orf_pol(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, L, pol):

        x_values = np.linspace(0, pi, 100)
        y_values = np.linspace(0, 2*pi, 100)
        X, Y = np.meshgrid(x_values,y_values) 
        f_values = Response.integrand(X, Y, psi, c1, xA1, xB1, c2, xA2, xB2, c, f, L, pol)

        gamma_x = numpy.trapz(f_values, x_values.reshape(1, 100, 1), axis=1)
        gamma = numpy.trapz(gamma_x, y_values.reshape(1, 1, 100))

        gamma = gamma.reshape(len(f))
        return gamma

    @staticmethod
    def overlap(det1, det2, f, psi, pol, shift_angle=False):
    
        c1, xA1, xB1, l1, which_det1 = det.detector(det1, shift_angle)
        c2, xA2, xB2, l2, which_det2 = det.detector(det2, shift_angle)
     
        result = Response.orf_pol(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, l1, pol)

        if(det1=='LISA 1' or det1=='LISA 2' or det1=='LISA 3'):
            return (2/5*result)
        else:
            return np.array(result)
        
