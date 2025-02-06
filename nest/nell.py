import numpy as np
from numpy import cos, sin, pi, sqrt, arctan2
from nest import detectors as det

from scipy.integrate import dblquad

from mpmath import mp, mpc, quad

import numpy as np
from scipy.integrate import simpson



from nest.skymap import AngularPatternFunction
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import mpmath as mp

from scipy.special import lpmv


from nest.utils import c, H0, h


# angular overlap redunction function - angular response

class AngularResponse:



    def Rellm_integrand(l, m, x, y, psi, c1, u1, v1, c2, u2, v2, c, f, pol, L):
        
        '''
        Integrand of the anisotropic response function
        # refs: Bartolo et al. 2022 
        '''

        f = f.reshape(len(f), 1, 1)
        
        F1 = AngularPatternFunction.F(x, y, psi, c1, u1, v1, f, L)
        F2 = AngularPatternFunction.F(x, y, psi, c2, u2, v2, f, L)

        sph_harm_val = sph_harm(m, l, y, x)

        if pol == 't':
            return (5 / (8 * pi)) * (F1[0] * np.conj(F2[0]) + F1[1] * np.conj(F2[1])) * sph_harm_val * sqrt(4 * pi) * sin(x)
        elif pol == 'v':
            return (5 / (8 * pi)) * (F1[2] * np.conj(F2[2]) + F1[3] * np.conj(F2[3])) * sph_harm_val * sqrt(4 * pi) * sin(x)
        elif pol == 's':
            return (15 / (4 * pi)) * (F1[4] * np.conj(F2[4])) * sph_harm_val * sqrt(4 * pi) * sin(x)
        else:
            raise ValueError('Unknown polarization')

    def Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L):
        
        '''
        Integral of the anisotropic response function
        '''

        x_values = np.linspace(0, pi, 100)
        y_values = np.linspace(0, 2*pi, 100)
        X, Y = np.meshgrid(x_values,y_values) 

        f_values = AngularResponse.Rellm_integrand(l, m, X, Y, psi, c1, u1, v1, c2, u2, v2, c, f, pol, L)

        gamma_x = np.trapz(f_values, x_values.reshape(1, 100, 1), axis=1)
        gamma = np.trapz(gamma_x, y_values.reshape(1, 1, 100))

        real_part = np.array([mp.mpf(x.real) for row in gamma for x in row])
        imag_part = np.array([mp.mpf(x.imag) for row in gamma for x in row])
        real_part = np.array(real_part, dtype=np.float64)
        imag_part = np.array(imag_part, dtype=np.float64)
   
        return real_part + 1j*imag_part


    def R_ell_func(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
        
        '''
        l dependent anisotropic response function
        '''

        m_values = np.arange(-l, l+1)
        total = 0
        psi = 0 
        for m in m_values:
            total += np.abs(AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L))**2
        return np.sqrt(total)


    def R_ell(l, det1, det2, f, pol, shift_angle=False):
        
        '''
        l = multipole order (int)
        det1, det2: detectors (string)
        f: frequency array (array float)
        psi: polarization angle (float)
        pol: polarization mode (string)
        shift_angle: shift angle between detectors (None or float)
        '''

        ec1, u1, v1, l1, _ = det.detector(det1, shift_angle)
        ec2, u2, v2, l2, _ = det.detector(det2, shift_angle)

        return AngularResponse.R_ell_func(l, ec1, u1, v1, ec2, u2, v2, c, f, pol, l1)




# LISA

    def R_ell_AET(l, channel, pol, f):

        '''
        l = multipole order (int)
        channel: channel in the AET basis (string)
        pol: polarization mode (string)
        f: frequency array (array float)
        '''

        c1, u1, v1, L, _ = det.detector('LISA 1', shift_angle=None)
        c2, u2, v2, L, _ = det.detector('LISA 2', shift_angle=None)

        if l % 2 == 0:

            def R_AA_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
                m_values = np.arange(-l, l+1)
                total = 0
                psi = 0 
                for m in m_values:
                    total += np.abs((1 + np.exp(-4j*np.pi*m/3))*AngularResponse.Rellm(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                    - 2*AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L)) **2
                return np.sqrt(total/4)

            def R_TT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
                m_values = np.arange(-l, l+1)
                total = 0
                psi = 0 
                for m in m_values:
                    total += (1 + 2*np.cos(2*np.pi*m/3))**2 * np.abs(AngularResponse.Rellm(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                                                + 2*AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L) )**2
                return np.sqrt(np.real(total)/9)

            def R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
                m_values = np.arange(-l, l+1)
                total = 0
                psi = 0 
                for m in m_values:
                    total += np.sin(np.pi*m/3)**2 * np.abs((1 + np.exp(2j*np.pi*m/3))*AngularResponse.Rellm(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                    - 2*AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L)) **2
                return np.sqrt(total/3)

            def R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
                m_values = np.arange(-l, l+1)
                total = 0
                psi = 0 
                for m in m_values:
                    total += np.sin(np.pi*m/3)**2 * np.abs((1 + np.exp(2j*np.pi*m/3))*AngularResponse.Rellm(l, m, u1, v1, c1, u1, v1, c1, psi, f, pol, L)
                                    + AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L)) **2
                return np.sqrt(2*total/3)
            
            if channel == 'AA' or channel=='EE':
                return 2/5*R_AA_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
            elif channel == 'TT':
                return 2/5*R_TT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
            elif channel == 'AE':
                return 2/5*R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
            elif channel == 'AT' or channel == 'ET':
                return 2/5*R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
            else:
                raise ValueError('Unknown channel')
            
        else:

            def R_AA_ell(f):
                return np.zeros(len(f))
            
            def R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
                print('starting AE')
                m_values = np.arange(-l, l+1)
                total = 0
                psi = 0 
                for m in m_values:
                    print(m)
                    total += (1 + 2*np.cos(2*np.pi*m/3))**2 * (np.abs(AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L) ))**2
                return np.sqrt(total/3) 
            
            def R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L):
                print('starting AT')
                m_values = np.arange(-l, l+1)
                total = 0
                psi = 0 
                for m in m_values:
                    print(m)
                    total += np.sin(np.pi*m/3)**2 * (np.abs(AngularResponse.Rellm(l, m, u1, v1, c1, u2, v2, c2, psi, f, pol, L) ))**2
                return np.sqrt(2*total)
            
            if channel == 'AA' or channel == 'EE' or channel == 'TT':
                print(channel)
                return 2/5*R_AA_ell(f)
            elif channel == 'AE':
                print(channel)
                return 2/5*R_AE_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
            elif channel == 'AT' or channel=='ET':
                print(channel)
                return 2/5*R_AT_ell(l, c1, u1, v1, c2, u2, v2, c, f, pol, L)
            else:
                raise ValueError('Unknown channel')


# Bartolo et al. 2022 eq.4.42 - 4.43

class Sensitivity_ell:

    def Omega_ell(det1, det2, Rl, f):

        '''
        det1, det2: detectors (string)
        Rl: anisotropic response function (array float)
        f: frequency array (array float)
        '''

        fi, PnI = det.detector_Pn(det1)
        fj, PnJ = det.detector_Pn(det2)

        Pni = np.interp(f, fi, PnI)
        Pnj = np.interp(f, fj, PnJ)

        Nl = 10 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(Pni * Pnj) / Rl

        return Nl
    


    def Omega_ell_LISA(f, l, pol):

        '''
        f: frequency array (array float)
        l: multipole order (int)
        pol: polarization mode (string)
        '''

        if l == 0:

            Rl_AA = AngularResponse.R_ell_AET(l, 'AA', pol, f)
            Rl_EE = Rl_AA
            Rl_TT =  AngularResponse.R_ell_AET(l, 'TT', pol, f)

            psd_A = det.LISA_noise_AET(f, 'A')
            psd_E = det.LISA_noise_AET(f, 'E')
            psd_T = det.LISA_noise_AET(f, 'T')

            Nl_AA = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_A) / Rl_AA
            Nl_EE = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_E) / Rl_EE
            Nl_TT = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_T * psd_T) / Rl_TT

            Nl = np.array([Nl_AA, Nl_EE, Nl_TT])

            return np.sum(1/(Nl)**2, axis=0)**(-0.5)
        
        
        elif l % 2 == 0 and l != 0:

            Rl_AA = AngularResponse.R_ell_AET(l, 'AA', pol, f) 
            Rl_EE = Rl_AA
            Rl_TT =  AngularResponse.R_ell_AET(l, 'TT', pol, f) 
            Rl_AE =  AngularResponse.R_ell_AET(l, 'AE', pol, f) 
            Rl_AT =  AngularResponse.R_ell_AET(l, 'AT', pol, f) 
            Rl_ET = Rl_AT

            psd_A = det.LISA_noise_AET(f, 'A')
            psd_E = det.LISA_noise_AET(f, 'E')
            psd_T = det.LISA_noise_AET(f, 'T')

            
            Nl_AA = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_A) / Rl_AA
            Nl_EE = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_E) / Rl_EE
            Nl_TT = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_T * psd_T) / Rl_TT
            Nl_AE = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_E) / Rl_AE
            Nl_AT = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_T) / Rl_AT
            Nl_ET = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_T) / Rl_ET

            Nl = np.array([Nl_AA, Nl_EE, Nl_TT, Nl_AE, Nl_AT, Nl_ET])

            return np.sum(1/(Nl)**2, axis=0)**(-0.5)


        else:
            
            print('starting l odd')
            Rl_AE =  AngularResponse.R_ell_AET(l, 'AE', pol, f) 
            Rl_AT =  AngularResponse.R_ell_AET(l, 'AT', pol, f) 
            Rl_ET =  Rl_AT

            psd_A = det.LISA_noise_AET(f, 'A')
            psd_E = det.LISA_noise_AET(f, 'E')
            psd_T = det.LISA_noise_AET(f, 'T')
            
            Nl_AE = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_E) / Rl_AE
            Nl_AT = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_T) / Rl_AT
            Nl_ET = 4 * np.pi**2 * np.sqrt(4*np.pi) / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_T) / Rl_ET

            Nl = np.array([Nl_AT, Nl_AE, Nl_ET])

            return (np.sum(1/(Nl)**2, axis=0))**(-0.5)
        


def PLS_l(det1, det2, Rl, f, fref, snr, Tobs, beta_min, beta_max, Cl, pol, shift_angle):

    Omega_eff_l = Sensitivity_ell.Omega_ell(det1, det2, Rl, f)/np.sqrt(4*np.pi)

    def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
        Tobs = Tobs * 365 * 24 * 3600
        integrand = (((f/fref)**(beta)) / (Omega_eff_l))**2 * Cl
        integral = np.trapz(integrand, f)
        return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
    
    def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
        return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * ((f/fref)**(beta))
    
    def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l):
        beta = np.linspace(beta_min, beta_max, 1000)
        Omega = []
        for i in range(len(beta)):
            Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], Omega_eff_l))     
        return beta, np.array(Omega)
    
    beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l)

    pls_l = np.zeros(len(f))

    for i in range(len(f)):
        pls_l[i] = np.max(Omega[:,i])

    return pls_l
    

def PLS_l_LISA(Omega_eff_l, f, fref, snr, Tobs, beta_min, beta_max, Cl, pol, shift_angle):

    #prova a calcolarlo per i singoli canali e poi metti tutto insieme

    def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
        Tobs = Tobs * 365 * 24 * 3600
        integrand = (((f/fref)**(beta)) / (Omega_eff_l))**2 * Cl
        integral = np.trapz(integrand, f)
        return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
    
    def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
        return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * ((f/fref)**(beta))
    
    def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l):
        beta = np.linspace(beta_min, beta_max, 1000)
        Omega = []
        for i in range(len(beta)):
            Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], Omega_eff_l))     
        return beta, np.array(Omega)
    
    beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l)

    pls_l = np.zeros(len(f))

    for i in range(len(f)):
        pls_l[i] = np.max(Omega[:,i])

    return pls_l


def Omega_ell_LISA(f, l, pol, fref, snr, Tobs, beta_min, beta_max, Cl):

    '''
    f: frequency array (array float)
    l: multipole order (int)
    pol: polarization mode (string)
    '''

    if l == 0:

        Rl_AA = AngularResponse.R_ell_AET(l, 'AA', pol, f)
        Rl_EE = Rl_AA
        Rl_TT =  AngularResponse.R_ell_AET(l, 'TT', pol, f)

        psd_A = det.LISA_noise_AET(f, 'A')
        psd_E = det.LISA_noise_AET(f, 'E')
        psd_T = det.LISA_noise_AET(f, 'T')

        Nl_AA = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_A) / Rl_AA
        Nl_EE = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_E) / Rl_EE
        Nl_TT = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_T * psd_T) / Rl_TT

        Nl = np.array([Nl_AA, Nl_EE, Nl_TT])


        def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
            Tobs = Tobs * 365 * 24 * 3600
            integrand = (((f/fref)**(beta)) / (Omega_eff_l))**2 * Cl
            integral = np.trapz(integrand, f)
            return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
        
        def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
            return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * ((f/fref)**(beta))
        
        def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l):
            beta = np.linspace(beta_min, beta_max, 1000)
            Omega = []
            for i in range(len(beta)):
                Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], Omega_eff_l))     
            return beta, np.array(Omega)
        
        pls_l = np.zeros((len(Nl), len(f)))
        for i in range(len(Nl[:,0])):
            beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Nl[i,:])
            for j in range(len(f)):
                pls_l[i, j] = np.max(Omega[:,j])

        
        return np.sum(1/(pls_l)**2, axis=0)**(-0.5)


    elif l % 2 == 0 and l != 0:

        Rl_AA = AngularResponse.R_ell_AET(l, 'AA', pol, f)
        Rl_EE = Rl_AA
        Rl_TT =  AngularResponse.R_ell_AET(l, 'TT', pol, f)
        Rl_AE =  AngularResponse.R_ell_AET(l, 'AE', pol, f)
        Rl_AT =  AngularResponse.R_ell_AET(l, 'AT', pol, f)
        Rl_ET = Rl_AT

        psd_A = det.LISA_noise_AET(f, 'A')
        psd_E = det.LISA_noise_AET(f, 'E')
        psd_T = det.LISA_noise_AET(f, 'T')

        Nl_AA = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_A) / Rl_AA
        Nl_EE = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_E) / Rl_EE
        Nl_TT = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_T * psd_T) / Rl_TT
        Nl_AE = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_E) / Rl_AE
        Nl_AT = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_T) / Rl_AT
        Nl_ET = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_T) / Rl_ET

        Nl = np.array([Nl_AA, Nl_EE, Nl_TT, Nl_AE, Nl_AT, Nl_ET])


        def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
            Tobs = Tobs * 365 * 24 * 3600
            integrand = (((f/fref)**(beta)) / (Omega_eff_l))**2 * Cl
            integral = np.trapz(integrand, f)
            return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
        
        def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
            return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * ((f/fref)**(beta))
        
        def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l):
            beta = np.linspace(beta_min, beta_max, 1000)
            Omega = []
            for i in range(len(beta)):
                Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], Omega_eff_l))     
            return beta, np.array(Omega)
        
        pls_l = np.zeros((len(Nl), len(f)))
        for i in range(len(Nl[:,0])):
            beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Nl[i,:])
            for j in range(len(f)):
                pls_l[i, j] = np.max(Omega[:,j])

        
        return np.sum(1/(pls_l)**2, axis=0)**(-0.5)

    else:
        
        Rl_AE =  AngularResponse.R_ell_AET(l, 'AE', pol, f)
        Rl_AT =  AngularResponse.R_ell_AET(l, 'AT', pol, f)
        Rl_ET = Rl_AT

        psd_A = det.LISA_noise_AET(f, 'A')
        psd_E = det.LISA_noise_AET(f, 'E')
        psd_T = det.LISA_noise_AET(f, 'T')

        Nl_AE = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_E) / Rl_AE
        Nl_AT = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_A * psd_T) / Rl_AT
        Nl_ET = 4 * np.pi**2  / (3* (H0/h)**2) * f**3 * np.sqrt(psd_E * psd_T) / Rl_ET

        Nl = np.array([Nl_AE, Nl_AT, Nl_ET])


        def Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l):
            Tobs = Tobs * 365 * 24 * 3600
            integrand = (((f/fref)**(beta)) / (Omega_eff_l))**2 * Cl
            integral = np.trapz(integrand, f)
            return  snr /np.sqrt(2*Tobs)/np.sqrt(integral)
        
        def Omega_GW(f, fref, snr, Tobs, beta, Omega_eff_l):
            return Omega_beta(f, fref, snr, Tobs, beta, Omega_eff_l) * ((f/fref)**(beta))
        
        def all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Omega_eff_l):
            beta = np.linspace(beta_min, beta_max, 1000)
            Omega = []
            for i in range(len(beta)):
                Omega.append(Omega_GW(f, fref, snr, Tobs, beta[i], Omega_eff_l))     
            return beta, np.array(Omega)
        
        pls_l = np.zeros((len(Nl), len(f)))
        for i in range(len(Nl[:,0])):
            beta, Omega = all_Omega_GW(f, fref, snr, Tobs, beta_min, beta_max, Nl[i,:])
            for j in range(len(f)):
                pls_l[i, j] = np.max(Omega[:,j])

        
        return np.sum(1/(pls_l)**2, axis=0)**(-0.5)








    

