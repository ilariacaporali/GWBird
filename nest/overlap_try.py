import numpy as np
#import jax.numpy as np
from numpy import cos, sin, pi, sqrt, arctan2
from nest import detectors as det
from nest import pls
import matplotlib.pyplot as plt
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
from multiprocessing import Pool
from tqdm import tqdm
from numba import  jit
from scipy.integrate import simps
import mpmath as mp


plt.rcParams['figure.dpi'] = 200
REarth = 6.371 * 1e6 #m
c = 3*1e8

def rotate(x, beta):
    return np.array([x[0]*cos(beta) + x[1]*sin(beta), -x[0]*sin(beta) + x[1]*cos(beta), x[2]])

# basis

@jit
def u_v_Omega_basis(theta, phi):
    # Construct each component as a 2D array first
    u0 = np.cos(theta) * np.cos(phi)
    u1 = np.cos(theta) * np.sin(phi)
    u2 = -np.sin(theta)

    v0 = -np.sin(phi)
    v1 = np.cos(phi)
    v2 = np.zeros_like(phi)  # Ensures v has the same shape as u and Omega

    Omega0 = np.sin(theta) * np.cos(phi)
    Omega1 = np.sin(theta) * np.sin(phi)
    Omega2 = np.cos(theta)

    # Now, stack the components along a new axis to create 3D arrays
    u = np.stack((u0, u1, u2), axis=0)
    v = np.stack((v0, v1, v2), axis=0)
    Omega = np.stack((Omega0, Omega1, Omega2), axis=0)

    return u, v, Omega

def Omega(theta, phi):
    return u_v_Omega_basis(theta, phi)[2]

@jit
def m_n_Omega_basis(theta, phi, psi):
    u, v, Omega = u_v_Omega_basis(theta, phi)
    m = u*np.cos(psi) + v*np.sin(psi)
    n = -u*np.sin(psi) + v*np.cos(psi)
    return m, n, Omega

# polarization tensors

def e_pol(theta, phi, psi): #polarization tensors    
    m, n, Omega = m_n_Omega_basis(theta, phi, psi)
    e_plus = np.einsum('iab,jab->ijab', m, m)- np.einsum('iab,jab->ijab', n, n)
    e_cross = np.einsum('iab,jab->ijab',m, n) + np.einsum('iab,jab->ijab',n, m)
    e_x = np.einsum('iab,jab->ijab',m, Omega) + np.einsum('iab,jab->ijab',Omega, m)
    e_y = np.einsum('iab,jab->ijab',n, Omega) + np.einsum('iab,jab->ijab',Omega, n)
    e_b = np.einsum('iab,jab->ijab',m,m) + np.einsum('iab,jab->ijab',n,n)
    e_l = sqrt(2)*(np.einsum('iab,jab->ijab',Omega, Omega))
    return e_plus, e_cross, e_x, e_y, e_b, e_l


# transfer function

def transfer_function(L, e, f, theta, phi, psi):
    omega = m_n_Omega_basis(theta, phi, psi)[2]
    l = e
    #print(f.shape)
    f = f.reshape((len(f), 1, 1))
    inner = np.einsum('iab,i->ab', omega, l)
    #print(L)
    f_star = c/2/np.pi/L
    #print(f_star)
    return 1/2 * (  np.sinc(f/2/np.pi/f_star * (1-inner)) * np.exp(-1j*f/2/f_star * (3+inner))\
                                    + np.sinc(f/2/np.pi/f_star * (1+inner)) * np.exp(-1j*f/2/f_star * (1+inner)))


def get_tf(f, which_det, theta, phi, psi, shift_angle=None):
    c, u, v, l, det_name = det.detector(which_det, shift_angle)
    tf = transfer_function(l, u, f, theta, phi, psi)
    return tf


# detector tensor + angular pattern function


def F_transfer(theta, phi, psi, ce, e1, e2, f, L): #angular pattern function
    e_plus, e_cross, e_x, e_y, e_b, e_l = e_pol(theta, phi, psi)
    omega = m_n_Omega_basis(theta, phi, psi)[2]
    f = f.reshape(len(f), 1, 1)
    f_star = c/2/np.pi
    tr_arm1 = transfer_function(L, e1, f, theta, phi, psi)
    tr_arm2 = transfer_function(L, e2, f, theta, phi, psi)
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
  

# angular overlap redunction function - angular response

#      (overlap reduction function averaged over the sky)


def integrand_transfer(x, y, psi, c1, u1, v1, c2, u2, v2, c, f, L, pol):

    F1 = F_transfer(x, y, psi, c1, u1, v1, f, L)
    F2 = F_transfer(x, y, psi, c2, u2, v2, f, L)

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


def orf_pol(u1, v1, c1, l1, u2, v2, c2, l2, psi, f, L, pol):
    mp.dps = 50  # Imposta la precisione desiderata

    x_values = np.linspace(0, pi, 100)
    y_values = np.linspace(0, 2*pi, 100)
    X, Y = np.meshgrid(x_values,y_values) 
    f_values = integrand_transfer(X, Y, psi, c1, u1, v1, c2, u2, v2, c, f, L, pol)

    gamma_x = np.trapz(f_values, x_values.reshape(1, 100, 1), axis=1)
    gamma = np.trapz(gamma_x, y_values.reshape(1, 1, 100))

    real_part = np.array([mp.mpf(x.real) for row in gamma for x in row])
    imag_part = np.array([mp.mpf(x.imag) for row in gamma for x in row])

    # Converti gli array di mpf in array di float
    real_part = np.array(real_part, dtype=np.float64)
    imag_part = np.array(imag_part, dtype=np.float64)

    return real_part

def overlap(detector1, detector2, f, psi, pol, shift_angle=False):
  
    if(detector1 == 'ET L2' and detector2 != 'ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, None)
    
    elif(detector2 == 'ET L2' and detector1 != 'ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle)
        
    else:
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle=None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle=None)

    result = orf_pol(e11, e21, ec1, l1, e12, e22, ec2, l2, psi, f, l1, pol)

    if(detector1=='LISA 1' or detector1=='LISA 2' or detector1=='LISA 3'):
        return (2/5*result)
    else:
        return np.array(result)
    
