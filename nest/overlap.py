import numpy as np
from numpy import cos, sin, pi, sqrt
from nest import detectors as det
import matplotlib.pyplot as plt
import mpmath as mp

from nest.utils import c
from nest.skymap import AngularPatternFunction



# # angular overlap redunction function - angular response

# #      (overlap reduction function averaged over the sky)

class Response:

    def R_integrand(x, y, psi, c1, xA1, xB1, c2, xA2, xB2, c, f, L, pol):
        
        '''
        Integrand of the overlap reduction function
        '''

        F1 = AngularPatternFunction.F(x, y, psi, c1, xA1, xB1, f, L)
        F2 = AngularPatternFunction.F(x, y, psi, c2, xA2, xB2, f, L)

        f = f.reshape(len(f), 1, 1)

        if (pol == 't'):
            return (5/(8*pi))*\
                ( F1[0]* np.conj( F2[0]) \
                + F1[1] *np.conj(F2[1])) \
                *sin(x)
        
        elif (pol == 'v'):
            return (5/(8*pi))*\
                ( F1[2]* np.conj( F2[2]) \
                + F1[3] *np.conj(F2[3])) \
                *sin(x)
        
        elif (pol == 's'):
            return (15/(4*pi))*\
                ( F1[4]* np.conj( F2[4]) ) \
                * sin(x)
        
                   
        else:
            raise ValueError('Unknown polarization')


    def R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, L, pol):
        
        '''
        Overlap reduction function for a given polarization
        '''

        x_values = np.linspace(0, pi, 100)
        y_values = np.linspace(0, 2*pi, 100)
        X, Y = np.meshgrid(x_values,y_values) 
        f_values = Response.R_integrand(X, Y, psi, c1, xA1, xB1, c2, xA2, xB2, c, f, L, pol)
        gamma_x = np.trapz(f_values, x_values, axis=1)
        gamma = np.trapz(gamma_x, y_values)
        # real_part = np.array([mp.mpf(x.real) for row in gamma for x in row])
        # imag_part = np.array([mp.mpf(x.imag) for row in gamma for x in row])
        # real_part = np.array(real_part, dtype=np.float64)
        # imag_part = np.array(imag_part, dtype=np.float64)
        return np.real(gamma) #+ 1j*imag_part


    def overlap(det1, det2, f, psi, pol, shift_angle=False):
        
        '''
        det1, det2: detectors (string)
        f: frequency array (array float)
        psi: polarization angle (float)
        pol: polarization mode (string)
        shift_angle: shift angle between detectors (None or float)
        '''
    
        c1, xA1, xB1, l1, _ = det.detector(det1, shift_angle)
        c2, xA2, xB2, l2, _ = det.detector(det2, shift_angle)
     
        result = Response.R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, l1, pol)

        if(det1=='LISA 1' or det1=='LISA 2' or det1=='LISA 3'):
            return (2/5*result)
        else:
            return np.array(result)
        

    

    

        
    def overlap_AET(channel, f, psi, pol):
        
        '''
        channel: AA, EE or TT (string)
        f: frequency array (array float)
        psi: polarization angle (float)
        pol: polarization mode (string)
        shift_angle: shift angle between detectors (None or float)
        '''
    
        c1, xA1, xB1, l1, _ = det.detector('LISA 1', shift_angle=None)
        c2, xA2, xB2, l2, _ = det.detector('LISA 2', shift_angle=None)

        auto = Response.R_func(xA1, xB1, c1, l1, xA1, xB1, c1, l1, psi, f, l1, pol)
        cross = Response.R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, l1, pol)
     
        if channel=='AA' or channel=='EE':
            return 2/5*np.abs(auto-cross)
        if channel=='TT':
            return 2/5*np.abs(auto+2*cross)
        else:
            return print('Select AA, EE or TT')
        

def overlap_3pol(det1, det2, det3, f, pol, shift_angle):

    '''
    det1, det2, det3: detectors (string)
    f: frequency array (array float)
    fref: reference frequency (float)
    pol: polarization mode (string)
    snr: signal-to-noise ratio threshold (float)
    Tobs: observation time (float) - in YEARS
    beta_min, beta_max: range of beta values (float)
    shift_angle: shift angle (None or float)
    '''

    fi, PnI = det.detector_Pn(det1)
    fj, PnJ = det.detector_Pn(det2)
    fk, PnK = det.detector_Pn(det3)

    PnI = np.interp(f, fi, PnI)
    PnJ = np.interp(f, fj, PnJ)
    PnK = np.interp(f, fk, PnK)


    orf_12_t = Response.overlap(det1, det2, f, 0 , 't', shift_angle )
    orf_23_t = Response.overlap(det2, det3, f, 0 , 't', shift_angle )
    orf_31_t = Response.overlap(det1, det3, f, 0 , 't', shift_angle )

    orf_12_v = Response.overlap(det1, det2, f, 0 , 'v', shift_angle )
    orf_23_v = Response.overlap(det2, det3, f, 0 , 'v', shift_angle )
    orf_31_v = Response.overlap(det1, det3, f, 0 , 'v', shift_angle )

    orf_12_s = Response.overlap(det1, det2, f, 0 , 's', shift_angle )
    orf_23_s = Response.overlap(det2, det3, f, 0 , 's', shift_angle )
    orf_31_s = Response.overlap(det1, det3, f, 0 , 's', shift_angle )

    orfIJK = orf_12_t * ( orf_23_s * orf_31_v - orf_31_s * orf_23_v) + \
                orf_23_t * ( orf_31_s * orf_12_v - orf_12_s * orf_31_v) + \
                orf_31_t * ( orf_12_s * orf_23_v - orf_23_s * orf_12_v)
    
    a_1_t = orf_23_s * orf_31_v - orf_31_s * orf_23_v
    a_2_t = orf_31_s * orf_12_v - orf_12_s * orf_31_v
    a_3_t = orf_12_s * orf_23_v - orf_23_s * orf_12_v

    a_1_v = orf_23_s * orf_31_t - orf_31_s * orf_23_t
    a_2_v = orf_31_s * orf_12_t - orf_12_s * orf_31_t
    a_3_v = orf_12_s * orf_23_t - orf_23_s * orf_12_t

    a_1_s = orf_23_t * orf_31_v - orf_31_t * orf_23_v
    a_2_s = orf_31_t * orf_12_v - orf_12_t * orf_31_v   
    a_3_s = orf_12_t * orf_23_v - orf_23_t * orf_12_v

    a_1 = np.zeros(len(f))
    a_2 = np.zeros(len(f))
    a_3 = np.zeros(len(f))

    if pol == 't':
        a_1 = a_1_t
        a_2 = a_2_t
        a_3 = a_3_t

    elif pol == 'v':
        a_1 = a_1_v
        a_2 = a_2_v
        a_3 = a_3_v

    elif pol == 's':
        a_1 = a_1_s
        a_2 = a_2_s
        a_3 = a_3_s

    else:
        raise ValueError('Unknown polarization')
    

    def S_eff(orfIJK, a_1, a_2, a_3, Ni, Nj, Nk):
        den = a_1**2 * Ni * Nk + a_2**2 * Ni * Nj + a_3**2 * Nj * Nk
        return (orfIJK**2 / den)
    
    return S_eff(orfIJK, a_1, a_2, a_3, PnI, PnJ, PnK)
