import numpy as np
from numpy import sin, pi
from gwbird import detectors 
from gwbird.utils import c
from gwbird.skymap import AngularPatternFunction



# # angular overlap redunction function - angular response

# #      (overlap reduction function averaged over the sky)

class Response:

    def R_integrand(x, y, psi, c1, xA1, xB1, l1, c2, xA2, xB2, l2, f, pol):
        
        '''
        Integrand of the overlap reduction function

        x: polar angle (float)
        y: azimuthal angle (float)
        psi: polarization angle (float)
        c1: position of the first detector (array float)
        xA1: unit vector pointing towards the first detector first arm(array float)
        xB1: unit vector pointing towards the first detector second arm (array float)
        l1 : length of the arm of the first detector (float)
        c2: position of the second detector (array float)
        xA2: unit vector pointing towards the second detector first arm (array float)
        xB2: unit vector pointing towards the second detector second arm (array float)
        l2: length of the arm of the second detector (float
        f: frequency array (array float)
        pol: polarization mode (string)

        return: integrand of the overlap reduction function

        '''

        F1 = AngularPatternFunction.F(x, y, psi, c1, xA1, xB1, f, l1)
        F2 = AngularPatternFunction.F(x, y, psi, c2, xA2, xB2, f, l2)

        f = f.reshape(len(f), 1, 1)

        if (pol == 't'): # https://arxiv.org/pdf/1310.5300
            return (5/(8*pi))*\
                ( F1[0]* np.conj( F2[0]) \
                + F1[1] *np.conj(F2[1])) \
                *sin(x)
        
        elif (pol == 'v'): # https://arxiv.org/pdf/2105.13197
            return (5/(8*pi))*\
                ( F1[2]* np.conj( F2[2]) \
                + F1[3] *np.conj(F2[3])) \
                *sin(x)
        
        elif (pol == 's'): # https://arxiv.org/pdf/2105.13197
            return (15/(4*pi))*\
                ( F1[4]* np.conj( F2[4]) ) \
                * sin(x)
        
        elif (pol =='I'): # https://arxiv.org/pdf/0707.0535
            return (5/(8*pi))*\
                ( F1[0]* np.conj( F2[0]) \
                + F1[1] *np.conj(F2[1])) \
                *sin(x)
        
        elif(pol=='V'): # https://arxiv.org/pdf/0707.0535
            return 1j*(5/(8*pi))*\
                ( F1[0]* np.conj( F2[1]) \
                - F1[1] *np.conj(F2[0])) \
                *sin(x)
     
        else:
            raise ValueError('Unknown polarization')
        




    def R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, pol):
        
        '''
        Overlap reduction function for a given polarization

        xA1: unit vector pointing towards the first detector first arm(array float)
        xB1: unit vector pointing towards the first detector second arm (array float)
        c1: position of the first detector (array float)
        l1: length of the arm of the first detector (float)
        xA2: unit vector pointing towards the second detector first arm (array float)
        xB2: unit vector pointing towards the second detector second arm (array float)
        c2: position of the second detector (array float)
        l2: length of the arm of the second detector (float)
        psi: polarization angle (float)
        f: frequency array (array float)
        pol: polarization mode (string)

        return: overlap reduction function
        '''

        x_values = np.linspace(0, pi, 100)
        y_values = np.linspace(0, 2*pi, 100)
        X, Y = np.meshgrid(x_values,y_values)  
        f_values = Response.R_integrand(X, Y, psi, c1, xA1, xB1, l1, c2, xA2, xB2, l2, f, pol) # gamma values
        gamma_x = np.trapezoid(f_values, x_values, axis=1)
        gamma = np.trapezoid(gamma_x, y_values)
        return np.real(gamma) 
    


    def overlap(det1, det2, f, psi, pol, shift_angle=None):
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
        - pol: str (Polarization of the signal, 't' for tensor, 'v' for vector, 's' for scalar, 'I' for intensity, 'V' for circular)

        Optional parameters:
        - shift_angle: float (Shift the angle of the response if considering ET 2L in radians)
        """
        def get_detector_params(det):
            if isinstance(det, str):
                # Handle standard detectors
                if det in ['LISA A', 'LISA E', 'LISA T']:
                    cX, xAX, xBX, lX, _ = detectors.detector('LISA X', shift_angle=None)
                    cY, xAY, xBY, lY, _ = detectors.detector('LISA Y', shift_angle=None)
                    cZ, xAZ, xBZ, lZ, _ = detectors.detector('LISA Z', shift_angle=None)
                    
                    if det == 'LISA A':
                        return cX, xAX, xBX, lX, det
                    elif det == 'LISA E':
                        return cY, xAY, xBY, lY, det
                    elif det == 'LISA T':
                        return cZ, xAZ, xBZ, lZ, det
                elif det in ['ET A', 'ET E', 'ET T']:
                    cX, xAX, xBX, lX, _ = detectors.detector('ET X', shift_angle=None)
                    cY, xAY, xBY, lY, _ = detectors.detector('ET Y', shift_angle=None)
                    cZ, xAZ, xBZ, lZ, _ = detectors.detector('ET Z', shift_angle=None)
                    
                    if det == 'ET A':
                        return cX, xAX, xBX, lX, det
                    elif det == 'ET E':
                        return cY, xAY, xBY, lY, det
                    elif det == 'ET T':
                        return cZ, xAZ, xBZ, lZ, det
                else:
                    return detectors.detector(det, shift_angle)
            elif isinstance(det, list) and len(det) == 5:
                # Handle custom detectors
                c, xA, xB, l, name = det
                return np.array(c), np.array(xA), np.array(xB), l, name
            else:
                raise ValueError(f"Invalid detector format: {det}")

        # Handle standard detectors or custom detectors
        c1, xA1, xB1, l1, name1 = get_detector_params(det1)
        c2, xA2, xB2, l2, name2 = get_detector_params(det2)

        # Check if detectors are lists or strings and convert them to appropriate string names for the special_map
        if isinstance(det1, list):
            name1 = det1[4]  # Custom detectors have a name in the 5th element
        if isinstance(det2, list):
            name2 = det2[4]  # Custom detectors have a name in the 5th element

        # Special handling for LISA and ET cases
        special_map = {
            ('LISA A', 'LISA A'): ('LISA X', 'LISA Y', -1),
            ('LISA E', 'LISA E'): ('LISA X', 'LISA Y', -1),
            ('LISA T', 'LISA T'): ('LISA X', 'LISA Y', 2),
            ('ET A', 'ET A'): ('ET X', 'ET Y', -1),
            ('ET E', 'ET E'): ('ET X', 'ET Y', -1),
            ('ET T', 'ET T'): ('ET X', 'ET Y', 2)
        }

        # Check if det1 and det2 match any special combinations
        if (name1, name2) in special_map:
            det_x, det_y, factor = special_map[(name1, name2)]
            c1, xA1, xB1, l1, _ = detectors.detector(det_x, shift_angle=None)
            c2, xA2, xB2, l2, _ = detectors.detector(det_y, shift_angle=None)
            auto = Response.R_func(xA1, xB1, c1, l1, xA1, xB1, c1, l1, psi, f, pol)
            cross = Response.R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, pol)
            return auto + factor * cross

        # Check for invalid combinations
        special_channels = {'LISA A', 'LISA E', 'LISA T', 'ET A', 'ET E', 'ET T'}
        if (name1 in special_channels or name2 in special_channels) and (name1 != name2):
            raise ValueError("Put a valid combination of channels")
        
        
        # General case    
        return np.array(Response.R_func(xA1, xB1, c1, l1, xA2, xB2, c2, l2, psi, f, pol))
    

    def overlap_NANOGrav(f):

        '''

        Compute the overlap reduction function for a set of pulsars

        parameters:
        f: frequency array

        return:
        overlap: overlap reduction function for a set of pulsars

        '''


        def gamma_integrand(theta, phi, psi, p1, p2):
            Fp1 = AngularPatternFunction.F_pulsar(theta, phi, psi, p1)
            Fp2 = AngularPatternFunction.F_pulsar(theta, phi, psi, p2)
            gamma_ij = 3/ 2 * (Fp1[0] * Fp2[0] + Fp1[1] * Fp2[1])
            return gamma_ij 

        def gamma( p1, p2, f):
            theta = np.linspace(0, np.pi, 100)
            phi = np.linspace(0, 2*np.pi, 100)
            Theta, Phi = np.meshgrid(theta, phi)
            integrand = gamma_integrand(Theta, Phi, 0, p1, p2)
            integral = np.trapezoid(np.trapezoid(np.sin(Theta) * integrand, theta), phi)
            return np.abs(integral)/4* np.pi


        N_pulsar, pulsar_xyz, DIST_array = detectors.get_NANOGrav_pulsars()
        
        overlap = np.zeros(len(f))

        for i in range(N_pulsar):
            for j in range(i +1, N_pulsar):
                overlap += gamma( pulsar_xyz[i], pulsar_xyz[j], f)
                
        return overlap
        

        

            


            

