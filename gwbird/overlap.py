import numpy as np
from numpy import sin, pi
from gwbird import detectors as det
from gwbird.utils import c
from gwbird.skymap import AngularPatternFunction



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
        return np.real(gamma) 


    def overlap(det1, det2, f, psi, pol, shift_angle=False):
        
        '''
        det1, det2: detectors (string)
        f: frequency array (array float)
        psi: polarization angle (float)
        pol: polarization mode (string)
        shift_angle: shift angle between detectors (None or float)

        return: overlap reduction function
        '''
    
        if isinstance(det1, str):
            c1, xA1, xB1, l1, _ = det.detector(det1, shift_angle)
        else:
            c1, xA1, xB1, l1, _ = det1  # Se è già una lista con i parametri, li assegni direttamente

        if isinstance(det2, str):
            c2, xA2, xB2, l2, _ = det.detector(det2, shift_angle)
        else:
            c2, xA2, xB2, l2, _ = det2
     
        result = Response.R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, l1, pol)

        if(det1=='LISA 1' or det1=='LISA 2' or det1=='LISA 3'):
            return (2/5*result)
        else:
            return np.array(result)
        

    def available_detectors():
        '''
        List of available detectors
        '''
        return ['LIGO H', 'LIGO L', 'Virgo', 'KAGRA', 'CE', 'ET A', 'ET B', 'ET C', 'ET L1', 'ET L2', 'LISA 1', 'LISA 2', 'LISA 3']
        

        
    def overlap_AET(channel, f, psi, pol):
        
        '''
        channel: AA, EE or TT (string)
        f: frequency array (array float)
        psi: polarization angle (float)
        pol: polarization mode (string)
        shift_angle: shift angle between detectors (None or float)

        return: overlap reduction function in the AET configuration
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
        

