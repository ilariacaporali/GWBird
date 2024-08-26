import numpy as np
from numpy import pi, cos, sin, sqrt
import matplotlib.pyplot as plt
from pypack import detectors as det
from tqdm import tqdm
import matplotlib.pyplot as plt
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

# *** ingredients

c = 299792458 # speed of light

def u_v_Omega_basis(theta, phi):
    u = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    Omega = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return u, v, Omega

def Omega(theta, phi):
    return u_v_Omega_basis(theta, phi)[2]

def m_n_Omega_basis(theta, phi, psi):
    u, v, Omega = u_v_Omega_basis(theta, phi)
    m = u*np.cos(psi) + v*np.sin(psi)
    n = -u*np.sin(psi) + v*np.cos(psi)
    return m, n, Omega

def rotate(x, beta):
    return np.array([x[0]*cos(beta) + x[1]*sin(beta), -x[0]*sin(beta) + x[1]*cos(beta), x[2]])

def e_pol(theta, phi, psi): #polarization tensors    
    m, n, Omega = m_n_Omega_basis(theta, phi, psi)
    e_plus = np.einsum('i,j->ij', m, m)- np.einsum('i,j->ij', n, n)
    e_cross = np.einsum('i,j->ij',m, n) + np.einsum('i,j->ij',n, m)
    e_x = np.einsum('i,j->ij',m, Omega) + np.einsum('i,j->ij',Omega, m)
    e_y = np.einsum('i,j->ij',n, Omega) + np.einsum('i,j->ij',Omega, n)
    e_b = np.einsum('i,j->ij',m,m) + np.einsum('i,j->ij',n,n)
    e_l = sqrt(2)*(np.einsum('i,j->ij',Omega, Omega))
    return e_plus, e_cross, e_x, e_y, e_b, e_l


def D_tensor(e1, e2): #detector tensor in the small antenna limit
    return 0.5*(np.outer(e1, e1)- np.outer(e2, e2))

# angular pattern function - antenna pattern

def F(theta, phi, psi, e1, e2): #angular pattern function
    e_plus, e_cross, e_x, e_y, e_b, e_l = e_pol(theta, phi, psi)
    D = D_tensor(e1, e2) #detector tensor with no transfer function
    F_plus = np.einsum('ij,ij->', D, e_plus)
    F_cross = np.einsum('ij,ij->', D, e_cross)
    F_x = np.einsum('ij,ij->', D, e_x)
    F_y = np.einsum('ij,ij->', D, e_y)
    F_b = np.einsum('ij,ij->', D, e_b)
    F_l = np.einsum('ij,ij->', D, e_l)
    return F_plus, F_cross, F_x, F_y, F_b, F_l 


def F_mod_tensor(theta, phi, psi, e1, e2): #modulus of the angular pattern function
    F_plus, F_cross = F(theta, phi, psi, e1, e2)[0], F(theta, phi, psi, e1, e2)[1]
    return sqrt(F_plus**2 + F_cross**2) # "a sort of mean among the two"

def F_mod_vector(theta, phi, psi, e1, e2): #modulus of the angular pattern function
    F_x, F_y = F(theta, phi, psi, e1, e2)[2], F(theta, phi, psi, e1, e2)[3]
    return sqrt(F_x**2 + F_y**2) # "a sort of mean among the two"

def F_mod_scalar(theta, phi, psi, e1, e2): #modulus of the angular pattern function
    F_b, F_l = F(theta, phi, psi, e1, e2)[4], F(theta, phi, psi, e1, e2)[5]
    return sqrt(F_b**2 + F_l**2) # "a sort of mean among the two"


# ******* recipe for the skymaps

def skymap_apf(psi, det_name1, nside, shift_angle=None):
         
        if(det_name1 =='ET L2'):
            ec1, u1, v1, l1, which_det1 = det.detector(det_name1, shift_angle)

        else :
            ec1, u1, v1, l1, which_det1 = det.detector(det_name1, shift_angle=None)

        npix = hp.nside2npix(nside)
        theta,phi = hp.pix2ang(nside, np.arange(npix))

        apf_t = np.zeros(len(theta), float)
        apf_v = np.zeros(len(theta), float)
        apf_s = np.zeros(len(theta), float)
        
        psi = 0 
        for i in tqdm(range(len(theta)), desc="Processing {0} ".format(which_det1)):
            F1 = F(theta[i], phi[i], psi, u1, v1)

            apf_t[i] = (F1[0]*np.conj(F1[0]) + F1[1]*np.conj(F1[1]))
            apf_v[i] = (F1[2]*np.conj(F1[2]) + F1[3]*np.conj(F1[3]))
            apf_s[i] = (F1[4]*np.conj(F1[4]) + F1[5]*np.conj(F1[5]))

        maps = [apf_t,
                apf_v,
                apf_s
                ]
        

        fig, ax = plt.subplots(1,3,figsize=(20, 4))
        ax = ax.flatten()
        plt.rcParams.update({'font.size': 17})
        titles = ["tensor", "vector", "scalar"]
        titles = np.array(titles).flatten()
        plt.suptitle(r"$F(\hat{{\Omega}})$ ({0})".format(which_det1))

        for i in range(len(maps)):
                plt.axes(ax[i])
                ax[i].set_title("F({0})".format(which_det1))
                hp.mollview(maps[i], 
                        flip="astro",
                        coord=["C"],
                        title=titles[i],
                        cmap='viridis',
                        min = '{:.2f}'.format(min(maps[i])),
                        max = '{:.2f}'.format(max(maps[i])),
                        notext=True,
                        hold=True)    
                hp.visufunc.graticule(dpar=45, dmer=60, coord='C', local=True, color='white', verbose=False, )       
        
        
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

        return maps

        
     


def skymap_overlap(psi, det_name1, det_name2, nside, f, shift_angle=None):
    
        if(det_name1 =='ET L2' and det_name2 !='ET L2'):
            ec1, u1, v1, l1, which_det1 = det.detector(det_name1, shift_angle)
            ec2, u2, v2, l2, which_det2 = det.detector(det_name2, None)
        
        elif(det_name2 =='ET L2' and det_name1 !='ET L2'):
            ec1, u1, v1, l1, which_det1 = det.detector(det_name1, None)
            ec2, u2, v2, l2, which_det2 = det.detector(det_name2, shift_angle)
            
        else :
            ec1, u1, v1, l1, which_det1 = det.detector(det_name1, shift_angle=None)
            ec2, u2, v2, l2, which_det2= det.detector(det_name2, shift_angle=None)
        
        d = ec1 - ec2
        npix = hp.nside2npix(nside)
        theta,phi = hp.pix2ang(nside, np.arange(npix))
        
        psi = 0 
        
        overlap_t = np.zeros(len(theta), float)
        overlap_v = np.zeros(len(theta), float)
        overlap_s = np.zeros(len(theta), float)
        
        
        for i in range(len(theta)):
            F1 = F(theta[i], phi[i], psi, u1, v1)
            F2 = F(theta[i], phi[i], psi, u2, v2)
            cos_term =  np.cos(2*pi*f*(d[0]*Omega(theta[i], phi[i])[0] + d[1]*Omega(theta[i], phi[i])[1] + d[2]*Omega(theta[i], phi[i])[2])/c)
            overlap_t[i] = 5/(8*np.pi)*(F1[0]*F2[0] + F1[1]*F2[1])* cos_term #* sin(theta[i])
            overlap_v[i] = 5/(8*np.pi)*(F1[2]*np.conj(F2[2]) + F1[3]*np.conj(F2[3]))* np.cos(2*pi*f*(d[0]*Omega(theta[i], phi[i])[0] + d[1]*Omega(theta[i], phi[i])[1] + d[2]*Omega(theta[i], phi[i])[2])/c)
            overlap_s[i] = 3/(2*np.pi)*(F1[4]*np.conj(F2[4]))* np.cos(2*pi*f*(d[0]*Omega(theta[i], phi[i])[0] + d[1]*Omega(theta[i], phi[i])[1] + d[2]*Omega(theta[i], phi[i])[2])/c)
            

        maps = [overlap_t,
                overlap_v,
                overlap_s
                ]
        
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
                        min = '{:.2f}'.format(min(maps[i])),
                        max = '{:.2f}'.format(max(maps[i])),
                        notext=True,
                        hold=True)    
                hp.visufunc.graticule(dpar=45, dmer=60, coord='C', local=True, color='white', verbose=False, )       
        
        
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
        
        
   