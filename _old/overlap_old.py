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


plt.rcParams['figure.dpi'] = 200
REarth = 6.371 * 1e6 #m
c = 3*1e8

def rotate(x, beta):
    return np.array([x[0]*cos(beta) + x[1]*sin(beta), -x[0]*sin(beta) + x[1]*cos(beta), x[2]])

# ************************************************************************************
# ******************************** vector basis **************************************
# ************************************************************************************


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


# ************************************************************************************
# ************************* polarization tensors *************************************
# ************************************************************************************


def e_pol(theta, phi, psi): #polarization tensors    
    m, n, Omega = m_n_Omega_basis(theta, phi, psi)
    e_plus = np.einsum('iab,jab->ijab', m, m)- np.einsum('iab,jab->ijab', n, n)
    e_cross = np.einsum('iab,jab->ijab',m, n) + np.einsum('iab,jab->ijab',n, m)
    e_x = np.einsum('iab,jab->ijab',m, Omega) + np.einsum('iab,jab->ijab',Omega, m)
    e_y = np.einsum('iab,jab->ijab',n, Omega) + np.einsum('iab,jab->ijab',Omega, n)
    e_b = np.einsum('iab,jab->ijab',m,m) + np.einsum('iab,jab->ijab',n,n)
    e_l = sqrt(2)*(np.einsum('iab,jab->ijab',Omega, Omega))
    return e_plus, e_cross, e_x, e_y, e_b, e_l


# ************************************************************************************
# ************************* transfer function ****************************************
# ************************************************************************************


def transfer_function(L, e, f, theta, phi, psi):
    #transfer function for LISA
    omega = - m_n_Omega_basis(theta, phi, psi)[2]
    #print(omega.shape)
    l = e
    #print(l.shape)
    exp1 = np.exp(-1j*pi*f*L/c*(1+np.einsum('iab,i->ab', omega, l)))
    exp3 = np.exp(1j*pi*f*L/c*(3+np.einsum('iab,i->ab', omega, l)))
    sinc1 = np.sinc(pi*f*L/c*(1+np.einsum('iab,i->ab', omega, l)))
    sinc3 = np.sinc(pi*f*L/c*(1-np.einsum('iab,i->ab', omega, l)))
    return 0.5*(exp1*sinc1 + exp3*sinc3)

def get_tf(f, which_det, theta, phi, psi, shift_angle=None):
    c, u, v, l, det_name = det.detector(which_det, shift_angle)
    tf = transfer_function(l, u, f, theta, phi, psi)
    return tf


# ************************************************************************************
# *********************************** detector tensor ********************************
# ************************************************************************************


def D_tensor(e1, e2): #detector tensor in the small antenna limit
    return 0.5*(np.outer(e1, e1)- np.outer(e2, e2))

def D_tensor_transfer(e1, e2, L, f, theta, phi, psi): #detector tensor
    #L = 2.5e9
    tf1 = transfer_function(L, e1, f, theta, phi, psi)
    tf2 = transfer_function(L, e2, f, theta, phi, psi)
    out1 = np.einsum('i,j-> ij',e1, e1)
    out2 = np.einsum('i,j-> ij',e2, e2)
    p1 = np.einsum('ij, ab->ijab', out1, tf1)
    p2 = np.einsum('ij, ab->ijab', out2, tf2)
    D = 0.5*( p1 - p2)
    return D


# ************************************************************************************
# ****************** angular pattern function - antenna pattern **********************
# ************************************************************************************


def F(theta, phi, psi, e1, e2): #angular pattern function
    e_plus, e_cross, e_x, e_y, e_b, e_l = e_pol(theta, phi, psi)
    D = D_tensor(e1, e2) #detector tensor with no transfer function
    F_plus = np.einsum('ij,ijab->ab', D, e_plus)
    F_cross = np.einsum('ij,ijab->ab', D, e_cross)
    F_x = np.einsum('ij,ijab->ab', D, e_x)
    F_y = np.einsum('ij,ijab->ab', D, e_y)
    F_b = np.einsum('ij,ijab->ab', D, e_b)
    F_l = np.einsum('ij,ijab->ab', D, e_l)
    return F_plus, F_cross, F_x, F_y, F_b, F_l 

def F_transfer(theta, phi, psi, e1, e2, f, L): #angular pattern function
    e_plus, e_cross, e_x, e_y, e_b, e_l = e_pol(theta, phi, psi)
    D = D_tensor_transfer(e1, e2, L, f, theta, phi, psi)  #detector tensor with transfer function
    F_plus = np.einsum('ijab,ijab->ab', D, e_plus)
    F_cross = np.einsum('ijab,ijab->ab', D, e_cross)
    F_x = np.einsum('ijab,ijab->ab', D, e_x)
    F_y = np.einsum('ijab,ijab->ab', D, e_y)
    F_b = np.einsum('ijab,ijab->ab', D, e_b)
    F_l = np.einsum('ijab,ijab->ab', D, e_l)
    return F_plus, F_cross, F_x, F_y, F_b, F_l 




# ************************************************************************************
# *****************   Overlap Reduction Function (averaged)  *************************
# ************************************************************************************


# overlap reduction function averaged over the sky


def integrand(x, y, psi, c1, u1, v1, c2, u2, v2, c, f, L, pol):
    F1 = F(x, y, psi, u1, v1)
    F2 = F(x, y, psi, u2, v2)
    d = c1 - c2
    if (pol == 't'):
        return (5/(8*pi))*(F1[0]*F2[0] + F1[1]*F2[1]) * np.cos(2*pi*f*(d[0]*Omega(x, y)[0] + d[1]*Omega(x, y)[1] + d[2]*Omega(x, y)[2])/c)*sin(x)
    elif (pol == 'v'):
        return (5/(8*pi))*(F1[2]*F2[2] + F1[3]*F2[3]) * np.cos(2*pi*f*(d[0]*Omega(x, y)[0] + d[1]*Omega(x, y)[1] + d[2]*Omega(x, y)[2])/c)*sin(x)
    elif (pol == 's'):
        return (15/(4*pi))*(F1[4]*F2[4]) * np.cos(2*pi*f*(d[0]*Omega(x, y)[0] + d[1]*Omega(x, y)[1] + d[2]*Omega(x, y)[2])/c)*sin(x)


def orf_pol(u1, v1, c1, l1, u2, v2, c2, l2, psi, f, L, pol):
    x_values = np.linspace(0, pi, 100)
    y_values = np.linspace(0, 2*pi, 100)
    X, Y = np.meshgrid(x_values,y_values) 
    f_values = integrand(X, Y, psi, c1, u1, v1, c2, u2, v2, c, f, L, pol)
    gamma_x = simps(f_values, x_values, axis=0)
    gamma = simps(gamma_x, y_values)
    return gamma

def integrand_transfer(x, y, psi, c1, u1, v1, c2, u2, v2, c, f, L, pol):
    F1 = F_transfer(x, y, psi, u1, v1, f, L)
    F2 = F_transfer(x, y, psi, u2, v2, f, L)
    d = c1 - c2
    if (pol == 't'):
        return (1/(4*pi))*(F1[0]*F2[0] + F1[1]*F2[1]) * np.cos(2*pi*f*(d[0]*Omega(x, y)[0] + d[1]*Omega(x, y)[1] + d[2]*Omega(x, y)[2])/c)*sin(x)
    elif (pol == 'v'):
        return (1/(4*pi))*(F1[2]*F2[2] + F1[3]*F2[3]) * np.cos(2*pi*f*(d[0]*Omega(x, y)[0] + d[1]*Omega(x, y)[1] + d[2]*Omega(x, y)[2])/c)*sin(x)
    elif (pol == 's'): #fix normalization factor
        return (3/(2*pi))*(F1[4]*F2[4]) * np.cos(2*pi*f*(d[0]*Omega(x, y)[0] + d[1]*Omega(x, y)[1] + d[2]*Omega(x, y)[2])/c)*sin(x)

def orf_pol_transfer(u1, v1, c1, l1, u2, v2, c2, l2, psi, f, L, pol):
    x_values = np.linspace(0, pi, 100)
    y_values = np.linspace(0, 2*pi, 100)
    X, Y = np.meshgrid(x_values,y_values) 
    f_values = integrand_transfer(X, Y, psi, c1, u1, v1, c2, u2, v2, c, f, l1, pol)
    gamma_x = simps(f_values, x_values, axis=0)
    gamma = simps(gamma_x, y_values)
    return gamma

def overlap_worker(args):
    u1, v1, c1, l1, u2, v2, c2, l2, psi, f, L, pol = args
    return orf_pol(u1, v1, c1, l1, u2, v2, c2, l2, psi, f, L, pol)

def overlap_worker_transfer(args):
    u1, v1, c1, l1, u2, v2, c2, l2, psi, f, L, pol = args
    return orf_pol_transfer(u1, v1, c1, l1, u2, v2, c2, l2, psi, f, L, pol)

def overlap(detector1, detector2, f, psi, pol, shift_angle=False):
  
    if(detector1 =='ET L2' and detector2 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, None)
    
    elif(detector2 =='ET L2' and detector1 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle)
        
    else :
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle=None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle=None)
        
  
    args_list = [(e11, e21, ec1, l1, e12, e22, ec2, l2, psi, f[i], 0, pol) for i in range(len(f))]
    with Pool() as pool:
        orf = []
        for result in tqdm(pool.imap(overlap_worker, args_list), total=len(args_list), miniters=1):
            orf.append(result)
    if(pol == 't'):
        return np.array(orf)
    elif(pol == 's'):
        return np.array(orf)
    elif(pol == 'v'):
        return np.array(orf)
    
def overlap_transfer(detector1, detector2, f, psi, pol, shift_angle=False):
  
    if(detector1 =='ET L2' and detector2 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, None)
    
    elif(detector2 =='ET L2' and detector1 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle)
        
    else :
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle=None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle=None)
    
  
    args_list = [(e11, e21, ec1, l1, e12, e22, ec2, l2, psi, f[i], 0, pol) for i in range(len(f))]
    with Pool() as pool:
        orf = []
        for result in tqdm(pool.imap(overlap_worker_transfer, args_list), total=len(args_list), miniters=1):
            orf.append(result)
    if(pol == 't'):
        return orf
    elif(pol == 's'):
        return orf
    elif(pol == 'v'):
        return orf
    
def autocorr_norm(u1, v1, c1, l1, l2, psi, pol):
    if(pol == 't'):
        return orf_pol(u1, v1, c1, l1, u1, v1, c1, l2, psi, 0, 0, 't')
    if(pol == 'v'):
        return orf_pol(u1, v1, c1, l1, u1, v1, c1, l2, psi, 0, 0, 'v')
    if(pol == 's'):
        return orf_pol(u1, v1, c1, l1, u1, v1, c1, l2, psi, 0, 0, 's')


def overlap_all(detector1, detector2, f, psi, norm=False, shift_angle=False):
    if(detector1 =='ET L2' and detector2 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, None)

    elif(detector2 =='ET L2' and detector1 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle)
        
    else:
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle=None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle=None)

    args_list = [(e11, e21, ec1, l1, e12, e22, ec2, l2, psi, f[i], 0, 't') for i in range(len(f))]

    with Pool() as pool:
        orf_t = []
        for result in tqdm(pool.imap(overlap_worker, args_list), total=len(f), desc='Computing t-mode'):
            orf_t.append(result)

    args_list = [(e11, e21, ec1, l1, e12, e22, ec2, l2, psi, f[i], 0, 'v') for i in range(len(f))]
    with Pool() as pool:
        orf_v = []
        for result in tqdm(pool.imap(overlap_worker, args_list), total=len(f), desc='Computing v-mode'):
            orf_v.append(result)

    args_list = [(e11, e21, ec1, l1, e12, e22, ec2, l2, psi, f[i], 0, 's') for i in range(len(f))]
    with Pool() as pool:
        orf_s = []
        for result in tqdm(pool.imap(overlap_worker, args_list), total=len(f), desc='Computing s-mode'):
            orf_s.append(result)
            
    if(norm==False):
        return orf_t, orf_v, orf_s
    
    elif(norm==True):
        norm_t = autocorr_norm(e11, e21, ec1, l1, l2, psi, 't') 
        norm_v = autocorr_norm(e11, e21, ec1, l1, l2, psi, 'v')
        norm_s = autocorr_norm(e11, e21, ec1, l1, l2, psi, 's') 
        return orf_t/norm_t, orf_v/norm_v, orf_s/norm_s
    

def overlap_all_transfer(detector1, detector2, f, psi, norm=False, shift_angle=False):
    if(detector1 =='ET L2' and detector2 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, None)

    elif(detector2 =='ET L2' and detector1 !='ET L2'):
        ec1, e11, e21, l1, det1_name = det.detector(detector1, None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle)
        
    else:
        ec1, e11, e21, l1, det1_name = det.detector(detector1, shift_angle=None)
        ec2, e12, e22, l2, det2_name = det.detector(detector2, shift_angle=None)

    args_list = [(e11, e21, ec1, l1, e12, e22, ec2, l2, psi, f[i], 0, 't') for i in range(len(f))]

    with Pool() as pool:
        orf_t = []
        for result in tqdm(pool.imap(overlap_worker_transfer, args_list), total=len(f), desc='Computing t-mode'):
            orf_t.append(result)

    args_list = [(e11, e21, ec1, l1, e12, e22, ec2, l2, psi, f[i], 0, 'v') for i in range(len(f))]
    with Pool() as pool:
        orf_v = []
        for result in tqdm(pool.imap(overlap_worker_transfer, args_list), total=len(f), desc='Computing v-mode'):
            orf_v.append(result)

    args_list = [(e11, e21, ec1, l1, e12, e22, ec2, l2, psi, f[i], 0, 's') for i in range(len(f))]
    with Pool() as pool:
        orf_s = []
        for result in tqdm(pool.imap(overlap_worker_transfer, args_list), total=len(f), desc='Computing s-mode'):
            orf_s.append(result)
            
    if(norm==False):
        return orf_t, orf_v, orf_s
    
    elif(norm==True):
        norm_t = autocorr_norm(e11, e21, ec1, l1, l2, psi, 't') 
        norm_v = autocorr_norm(e11, e21, ec1, l1, l2, psi, 'v')
        norm_s = autocorr_norm(e11, e21, ec1, l1, l2, psi, 's') 
        return orf_t/norm_t, orf_v/norm_v, orf_s/norm_s
    

# orf output
    
def orf(detector1, detector2, f, psi, norm=False, shift_angle=False, transfer_function=False):
    if transfer_function == False:
        return overlap_all(detector1, detector2, f, psi, norm, shift_angle)
    elif transfer_function == True:
        return overlap_all_transfer(detector1, detector2, f, psi, norm, shift_angle)

def orf_p(detector1, detector2, f, psi, pol, norm=False, shift_angle=False, transfer_function=False):
    if transfer_function == False:
        return overlap(detector1, detector2, f, psi, pol, norm, shift_angle)
    elif transfer_function == True:
        return overlap_transfer(detector1, detector2, f, psi, pol, norm, shift_angle)



