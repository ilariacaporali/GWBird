import numpy as np
from numpy import cos, sin, pi, sqrt
from nest import detectors as det
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.integrate import simps

from nest.skymap import AngularPatternFunction



REarth = 6.371 * 1e6 #m
c = 299792458 # speed of light


# # angular overlap redunction function - angular response

# #      (overlap reduction function averaged over the sky)

class Response:
    @staticmethod
    def integrand(x, y, psi, c1, xA1, xB1, c2, xA2, xB2, c, f, L, pol):
        
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
        
        '''
        Overlap reduction function for a given polarization
        '''

        x_values = np.linspace(0, pi, 100)
        y_values = np.linspace(0, 2*pi, 100)
        X, Y = np.meshgrid(x_values,y_values) 
        f_values = Response.integrand(X, Y, psi, c1, xA1, xB1, c2, xA2, xB2, c, f, L, pol)

        gamma_x = simps(f_values, x_values, axis=1)
        gamma = simps(gamma_x, y_values)

        real_part = np.array([mp.mpf(x.real) for x in gamma])
        imag_part = np.array([mp.mpf(x.imag) for x in gamma])

        # Converti gli array di mpf in array di float
        real_part = np.array(real_part, dtype=np.float64)
        imag_part = np.array(imag_part, dtype=np.float64)

        return real_part + 1j*imag_part

    @staticmethod
    def overlap(det1, det2, f, psi, pol, shift_angle=False):
        
        '''
        det1, det2: detectors (string)
        f: frequency array (array float)
        psi: polarization angle (float)
        pol: polarization mode (string)
        shift_angle: shift angle between detectors (None or float)
        '''
    
        c1, xA1, xB1, l1, which_det1 = det.detector(det1, shift_angle)
        c2, xA2, xB2, l2, which_det2 = det.detector(det2, shift_angle)
     
        result = Response.orf_pol(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, l1, pol)

        if(det1=='LISA 1' or det1=='LISA 2' or det1=='LISA 3'):
            return (2/5*result)
        else:
            return np.array(result)
        
