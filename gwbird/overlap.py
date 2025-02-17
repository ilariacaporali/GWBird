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
        
    def R_integrand_RL(x, y, psi, c1, xA1, xB1, c2, xA2, xB2, c, f, L, pol):

        '''
        Integrand of the overlap reduction function in the RL configuration
        '''

        F1 = AngularPatternFunction.F_RL(x, y, psi, c1, xA1, xB1, f, L)
        F2 = AngularPatternFunction.F_RL(x, y, psi, c2, xA2, xB2, f, L)

        f = f.reshape(len(f), 1, 1)

        if (pol == 'R'):
            return (5/(8*pi))*\
                ( F1[0]* np.conj( F2[0])) \
                *sin(x)
        
        elif (pol == 'L'):
            return (5/(8*pi))*\
                ( F1[1]* np.conj( F2[1]) ) \
                *sin(x)
        
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
    
    def R_func_RL(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, L, pol):

        '''
        Overlap reduction function for a given polarization in the RL configuration
        '''

        x_values = np.linspace(0, pi, 100)
        y_values = np.linspace(0, 2*pi, 100)
        X, Y = np.meshgrid(x_values,y_values) 
        f_values = Response.R_integrand_RL(X, Y, psi, c1, xA1, xB1, c2, xA2, xB2, c, f, L, pol)
        gamma_x = np.trapz(f_values, x_values, axis=1)
        gamma = np.trapz(gamma_x, y_values)
        return np.real(gamma)


    def overlap(det1, det2, f, psi, pol, shift_angle=False):
        
        """
        Calculate the overlap response between two detectors.

        R = Response.overlap(det1, det2, f, psi, pol, shift_angle=False)

        Parameters:
        - det1, det2: str or list of str
            The name of the detector(s) to consider.
            The names must be in the list of detectors available in the response module.
            The list of available detectors can be obtained by calling the function detectors.available_detectors().
            The names of the detectors are case sensitive.
            If you want to provide a custom detector, you can provide the following information in a list:

            H = [c, xA, xB, l, name]

            - c: array_like of length 3 (Position of the detector in the Earth-centered frame in meters)
            - xA: array_like of length 3 (Unit vector pointing towards the detector in the Earth-centered frame)
            - xB: array_like of length 3 (Unit vector pointing towards the detector in the Earth-centered frame)
            - l: float (Length of the arm in meters)
            - name: str (Name of the detector)

        - f: array_like (Frequency in Hz)
        - psi: float (Polarization angle in radians)
        - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar)

        Optional parameters:
        - shift_angle: bool or float (Shift the angle of the response if considering ET 2L in radians)

        Example usage:
        """
    
        if isinstance(det1, str):
            c1, xA1, xB1, l1, _ = det.detector(det1, shift_angle)
        else:
            c1, xA1, xB1, l1, _ = det1  

        if isinstance(det2, str):
            c2, xA2, xB2, l2, _ = det.detector(det2, shift_angle)
        else:
            c2, xA2, xB2, l2, _ = det2
     
        result = Response.R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, l1, pol)

        if(det1=='LISA 1' or det1=='LISA 2' or det1=='LISA 3'):
            return (2/5*result)
        else:
            return np.array(result)
        
    def overlap_RL(det1, det2, f, psi, pol, shift_angle=False):

        '''
        det1, det2: detectors (string)
        f: frequency array (array float)
        psi: polarization angle (float)
        pol: polarization mode (string)
        shift_angle: shift angle between detectors (None or float)

        return: overlap reduction function in the RL configuration
        '''

        if isinstance(det1, str):
            c1, xA1, xB1, l1, _ = det.detector(det1, shift_angle)
        else:
            c1, xA1, xB1, l1, _ = det1

        if isinstance(det2, str):
            c2, xA2, xB2, l2, _ = det.detector(det2, shift_angle)
        else:
            c2, xA2, xB2, l2, _ = det2


        result = Response.R_func_RL(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, l1, pol)

        if(det1=='LISA 1' or det1=='LISA 2' or det1=='LISA 3'):
            return (2/5*result)
        
        else:
            return np.array(result)
        
    def overlap_IV(det1, det2, f, psi, pol, shift_angle=False):

        '''
        det1, det2: detectors (string)
        f: frequency array (array float)
        psi: polarization angle (float)
        pol: polarization mode (string)
        shift_angle: shift angle between detectors (None or float)

        return: overlap reduction function in the IV configuration
        '''

        if isinstance(det1, str):
            c1, xA1, xB1, l1, _ = det.detector(det1, shift_angle)
        else:
            c1, xA1, xB1, l1, _ = det1

        if isinstance(det2, str):
            c2, xA2, xB2, l2, _ = det.detector(det2, shift_angle)
        else:
            c2, xA2, xB2, l2, _ = det2

        R_L = Response.overlap_RL(det1, det2, f, psi, 'L')
        R_R = Response.overlap_RL(det1, det2, f, psi, 'R')

        if(pol=='I'):
            return (R_L+ R_R)/2
        if(pol=='V'):
            return (-R_L+R_R)/2


          

        
    def overlap_AET(channel, f, psi, pol):
        
        '''
        Evaluate the response function for LISA in the AET basis.

        Parameters:
        - channel : string (AA, EE or TT)
        - f: array_like (Frequency in Hz)
        - psi: float (Polarization angle in radians)
        - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar)

        return: array_like (overlap reduction function in the AET configuration)
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
        

    def overlap_AET_IV(channel, f, psi, pol):

        '''
        
        channel: AA, EE or TT (string)
        f: frequency array (array float)
        psi: polarization angle (float)
        pol: polarization mode (string)
        shift_angle: shift angle between detectors (None or float)
        
        return: overlap reduction function in the AET configuration
        '''

        auto = Response.overlap_IV('LISA 1', 'LISA 1', f, psi, pol, shift_angle=False)
        cross = Response.overlap_IV('LISA 1', 'LISA 2', f, psi, pol, shift_angle=False)

        if channel=='AA' or channel=='EE':
            return 2/5*np.abs(auto-cross)
        if channel=='TT':
            return 2/5*np.abs(auto+2*cross)
        else:
            return print('Select AA, EE or TT')
        


        

