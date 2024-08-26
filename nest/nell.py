import numpy as np
#import jax.numpy as np
from numpy import cos, sin, pi, sqrt, arctan2
from nest import detectors as det
from nest import pls
import matplotlib.pyplot as plt
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from numba import  jit
from scipy.integrate import simpson
from nest import overlap,  pls, skymap
from scipy.special import sph_harm
from mpmath import spherharm
from scipy.interpolate import interp1d  
from astropy.cosmology import Planck15

cosmo = Planck15
H0 =  cosmo.H0.to('1/s').value

c = 299792458 #m/s


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
    # Assicurarsi che f sia un array 2D con la stessa forma di theta e phi
    f = np.array(f).reshape((len(f), 1, 1))
    # Calcola la funzione di trasferimento
    omega = - m_n_Omega_basis(theta, phi, psi)[2]
    l = e
    exp3 = np.exp(-3j*pi*f*L/c*(1-np.einsum('iab,i->ab', omega, l)))
    exp1 = np.exp(-1j*pi*f*L/c*(1+np.einsum('iab,i->ab', omega, l)))
    sinc1 = np.sinc(f*L/c*(1-np.einsum('iab,i->ab', omega, l)))
    sinc3 = np.sinc(f*L/c*(1+np.einsum('iab,i->ab', omega, l)))
    return 0.5*(exp1*sinc1 + exp3*sinc3)

def get_tf(f, which_det, theta, phi, psi, shift_angle=None):
    c, u, v, l, det_name = det.detector(which_det, shift_angle)
    tf = transfer_function(l, u, f, theta, phi, psi)
    return tf

# detector tensor


def D_tensor_transfer(e1, e2, L, f, theta, phi, psi): #detector tensor
    tf1 = transfer_function(L, e1, f, theta, phi, psi)
    tf2 = transfer_function(L, e2, f, theta, phi, psi)
    out1 = np.einsum('i,j-> ij',e1, e1)
    out2 = np.einsum('i,j-> ij',e2, e2)
    p1 = np.einsum('ij, xab->ijxab', out1, tf1)
    p2 = np.einsum('ij, xab->ijxab', out2, tf2)
    D = 0.5*( p1 - p2)
    #D = tf1 - tf2
    return D

# angular patter function



def F_transfer(theta, phi, psi, e1, e2, f, L): #angular pattern function
    e_plus, e_cross, e_x, e_y, e_b, e_l = e_pol(theta, phi, psi)
    D = D_tensor_transfer(e1, e2, L, f, theta, phi, psi)  #detector tensor with transfer function
    F_plus = np.einsum('ijxab,ijab->xab', D, e_plus)
    F_cross = np.einsum('ijxab,ijab->xab', D, e_cross)
    F_x = np.einsum('ijxab,ijab->xab', D, e_x)
    F_y = np.einsum('ijxab,ijab->xab', D, e_y)
    F_b = np.einsum('ijxab,ijab->xab', D, e_b)
    F_l = np.einsum('ijxab,ijab->xab', D, e_l)
    return F_plus, F_cross, F_x, F_y, F_b, F_l 

# angular overlap redunction function 


#note: the part that takes into account the transfer function has been written with the LISA conventions
#       this means that instead of having a 5/8pi i have 1/4pi

def Rlm_integrand_transfer(l, m, x, y, psi, c1, u1, v1, c2, u2, v2, c, f, pol, L):
    F1 = F_transfer(x, y, psi, u1, v1, f, L)
    F2 = F_transfer(x, y, psi, u2, v2, f, L)
    d = c1 - c2
    omega = Omega(x, y)
    spherharm_vectorized = np.vectorize(spherharm)
    f = f.reshape(len(f), 1, 1)
    scal = np.einsum('i, ikl-> kl', d, omega)
    cos_term = np.cos(2*pi*f*(scal)/c)*sin(x)
    if (pol == 't'):
        return (5/(8*pi))*(F1[0]*np.conj(F2[0]) + F1[1]*np.conj(F2[1])) * cos_term * spherharm_vectorized(l, m, x, y)*np.sqrt(4*np.pi)
    elif (pol == 'v'):
        return (5/(8*pi))*(F1[2]*F2[2] + F1[3]*F2[3]) * cos_term * spherharm_vectorized(l, m, x, y)*np.sqrt(4*np.pi)
    elif (pol == 's'):
        return (15/(4*pi))*(F1[4]*F2[4]) * cos_term * spherharm_vectorized(l, m, x, y)*np.sqrt(4*np.pi)
    
def Rlm_transfer(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L):
    x_values = np.linspace(0, pi, 100)
    y_values = np.linspace(0, 2*pi, 100)
    X, Y = np.meshgrid(x_values,y_values) 
    f_values = Rlm_integrand_transfer(l, m, X, Y, psi, c1, u1, v1, c2, u2, v2, c, f, pol, L)
    #print(f_values.shape)
    gamma_x = np.trapz(f_values, x_values.reshape(1, 100, 1), axis=1)
    gamma = np.trapz(gamma_x, y_values.reshape(1, 1, 100))
    return gamma[0].real


def R_ell_transfer(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
    m_values = np.arange(-l, l+1)
    total = 0
    psi = 0 
    for m in m_values:
        total += np.abs(Rlm_transfer(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L))**2
    return np.sqrt(total)

def R_ell(l, det1, det2, f, pol, shift_angle):
    if(det1 =='ET L2' and det2 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(det1, shift_angle)
        ec2, e12, e22, l2, det2_name = det.detector(det2, None)

    elif(det2 =='ET L2' and det1 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(det1, None)
        ec2, e12, e22, l2, det2_name = det.detector(det2, shift_angle)
        
    else:
        ec1, e11, e21, l1, det1_name = det.detector(det1, shift_angle=None)
        ec2, e12, e22, l2, det2_name = det.detector(det2, shift_angle=None)

    return R_ell_transfer(l, ec1, e11, e21, ec2, e12, e22, c, f, pol, l1)

# N_ell sensitivity

def Omega_ell(det1, det2, Rl, f):

    fi, PnI = det.detector_Pn(det1)
    fj, PnJ = det.detector_Pn(det2)
    Nl = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* H0**2) * fi*3 * np.sqrt(PnI * PnJ) 

    Nl = np.interp(f, fi, Nl)

    return Nl/Rl/np.sqrt(4* np.pi)*0.49

# LISA

def R_AET_basis(l, channel, pol, f):
    
    c1, u1, v1, L, name = det.detector('LISA 1', shift_angle=None)
    c2, u2, v2, L, name = det.detector('LISA 2', shift_angle=None)

    if l % 2 == 0:

        def R_AA_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
            m_values = np.arange(-l, l+1)
            total = 0
            psi = 0 
            for m in m_values:
                total += np.abs((1 + np.exp(-4j*np.pi*m/3))*Rlm_transfer(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                - 2*Rlm_transfer(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L)) **2
            return np.sqrt(total/4)

        def R_TT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
            m_values = np.arange(-l, l+1)
            total = 0
            psi = 0 
            for m in m_values:
                total += (1 + np.cos(2*np.pi*m/3))**2 * np.abs(Rlm_transfer(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                                            + 2*Rlm_transfer(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L) )**2
            return np.sqrt(total/9)

        def R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
            m_values = np.arange(-l, l+1)
            total = 0
            psi = 0 
            for m in m_values:
                total += np.sin(np.pi*m/3)**2 * np.abs((1 + np.exp(-4j*np.pi*m/3))*Rlm_transfer(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                - 2*Rlm_transfer(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L)) **2
            return np.sqrt(total/3)

        def R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
            m_values = np.arange(-l, l+1)
            total = 0
            psi = 0 
            for m in m_values:
                total += np.sin(np.pi*m/3)**2 * np.abs((1 + np.exp(-4j*np.pi*m/3))*Rlm_transfer(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                + 2*Rlm_transfer(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L)) **2
            return np.sqrt(2*total/3)
        
        if channel == 'AA' or channel=='EE':
            return 2/5*R_AA_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
        elif channel == 'TT':
            return 2/5*R_TT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
        elif channel == 'AE':
            return 2/5*R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
        elif channel == 'AT' or 'ET':
            return 2/5*R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
        
    else:
        
        def R_AA_ell(f):
            return np.zeros(len(f))
        
        def R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
            m_values = np.arange(-l, l+1)
            total = 0
            psi = 0 
            for m in m_values:
                total += (1 + np.cos(2*np.pi*m/3**2))**2 * np.abs(Rlm_transfer(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L) )**2
            return np.sqrt(total/3) 
        
        def R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, Ll):
            m_values = np.arange(-l, l+1)
            total = 0
            psi = 0 
            for m in m_values:
                total += np.sin(np.pi*m/3)**2 * np.abs(Rlm_transfer(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L) )**2
            return np.sqrt(2*total)
        
        if channel == 'AA' or channel == 'EE' or channel == 'TT':
            return 2/5*R_AA_ell(f)
        elif channel == 'AE':
            return 2/5*R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
        elif channel == 'AT' or channel=='ET':
            return 2/5*R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)


# Bartolo et al. 2022 eq.4.43

def Omega_ell_LISA(f, l, pol):

    if l % 2 == 0:

        Rl_AA = R_AET_basis(l, 'AA', pol, f)
        Rl_TT =  R_AET_basis(l, 'TT', pol, f)

        Nl_AA = Nl_EE = Omega_ell('LISA 1', 'LISA 1', Rl_AA, f)
        Nl_TT = Omega_ell('LISA 1', 'LISA 1', Rl_TT, f)

        return np.sqrt(1 / ( 1/Nl_AA**2 + 1/Nl_EE**2 + 1/Nl_TT**2))
    
    else:
        
        Rl_AE =   R_AET_basis(l, 'AE', pol, f)
        Rl_AT =   R_AET_basis(l, 'AT', pol, f)
        
        Nl_AE = Omega_ell('LISA 1', 'LISA 1', Rl_AE, f)
        Nl_AT = Nl_ET = Omega_ell('LISA 1', 'LISA 1', Rl_AT, f)

        return np.sqrt(1 / ( 1/Nl_AE**2 + 1/Nl_AT**2 + 1/Nl_ET**2))



