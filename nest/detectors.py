import numpy as np
from numpy import pi, sin, cos
from scipy.spatial.transform import Rotation as R
from nest.utils import REarth

#***************************************************************************************************************

import numpy as np
from math import pi, sqrt

class Rotations:


    def rot_axis(vector, angle, axis):
        """
        Rotate a vector around a given axis by a certain angle
        """
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        return cos_theta * vector + sin_theta * np.cross(axis, vector) + (1 - cos_theta) * np.dot(axis, vector) * axis

    def rot_angle(vec1, vec2):
        """
        Calculate the angle between two vectors
        """
        return np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

    def find_perp(vec1, vec2):
        """
        Find a vector perpendicular to two given vectors
        """
        return np.cross(vec1, vec2)



class Observatories:

    # 2G detectors

    def LIGO_hanford(self):
        c = np.array([-0.33827472, -0.60015338, 0.72483525]) * REarth
        xA = np.array([-0.22389266154, 0.79983062746, 0.55690487831])
        xB = np.array([-0.91397818574, 0.02609403989, -0.40492342125])
        l = 4e3  # m
        return c, xA, xB, l, "LIGO Hanford"

    def LIGO_livingston(self):
        c = np.array([-0.01163537, -0.8609929, 0.50848387]) * REarth
        xA = np.array([-0.95457412153, -0.14158077340, -0.26218911324])
        xB = np.array([0.29774156894, -0.48791033647, -0.82054461286])
        l = 4e3  # m
        return c, xA, xB, l, "LIGO Livingston"

    def Virgo(self):
        c = np.array([0.71166465, 0.13195706, 0.69001505]) * REarth
        xA = np.array([-0.701, 0.201, 0.684])
        xB = np.array([-0.0485, -0.971, 0.236])
        l = 3e3  # m
        return c, xA, xB, l, "Virgo"

    def KAGRA(self):
        c = np.array([-0.59149285, 0.54570304, 0.59358605]) * REarth
        xA = np.array([-0.390, -0.838, 0.382])
        xB = np.array([0.706, -0.00580, 0.709])
        l = 3e3  # m
        return c, xA, xB, l, "KAGRA"
    
    # Cosmic Explorer in the location of LIGO Hanford

    def CE(self):
        c, xA, xB, l, _ = self.LIGO_hanford()
        return c, xA, xB, 4e4, "Cosmic Explorer"

    # Einstein Telescope (ET) - triangular configuration

    def ET_arms(self):
        xA = np.array([0., 0., 0.])
        xB = np.array([+1/2., sqrt(3)/2, 0])
        xC = np.array([-1/2, sqrt(3)/2, 0])
        lBA = xB - xA
        lCA = xC - xA
        lBC = xB - xC
        l = 1e4  # m
        return xA, xB, xC, lBA, lCA, lBC, l

    def ET_A(self):
        xA, xB, xC, lBA, lCA, lBC, l= self.ET_arms()
        return xA * l, lBA, lCA, l, "ET A"

    def ET_B(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.ET_arms()
        return xB * l, -lBC, -lBA, l, "ET B"

    def ET_C(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.ET_arms()
        return xC * l, -lCA, lBC, l, "ET C"
    
    # Einstein Telescope (ET) - L-shaped configuration

    def ET_L_sardinia(self):
        c = np.array([0.7499728, 0.12438134, 0.64966921]) * REarth
        xA = np.array([-0.639881, -0.106494, 0.761506])
        xB = Rotations.rot_axis(xA, pi/2, c)
        l = 1.5e4
        return c, xA, xB, l, "ET L-shaped Sardinia"

    def ET_L_netherlands(self, shift_angle=None):
        c1, xA1, xB1, l, _ = self.ET_L_sardinia()
        c2 = np.array([0.62969256, 0.06530073, 0.77409502]) * REarth

        xA2_int = Rotations.rot_axis(xA1, shift_angle, c1) if shift_angle else xA1
        xB2_int = Rotations.rot_axis(xB1, shift_angle, c1) if shift_angle else xB1

        beta = Rotations.rot_angle(c1, c2)
        rot_ax = Rotations.find_perp(c1, c2)
        xA2 = Rotations.rot_axis(xA2_int, beta, rot_ax)
        xB2 = Rotations.rot_axis(xB2_int, beta, rot_ax)
        return c2, xA2, xB2, l, "ET L-shaped Netherlands"


    # LISA (Laser Interferometer Space Antenna)

    def LISA_arms(self):
        l = 2.5e9  # m
        xA = np.array([0., 0., 0.])
        xB = np.array([+1/2., sqrt(3)/2, 0])
        xC = np.array([-1/2, sqrt(3)/2, 0])
        lBA = xB - xA
        lCA = xC - xA
        lBC = xB - xC
        return xA, xB, xC, lBA, lCA, lBC, l

    def LISA1(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.LISA_arms()
        return xA * l, lBA, lCA, l, "LISA 1"

    def LISA2(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.LISA_arms()
        return xB * l, -lBC, -lBA, l, "LISA 2"

    def LISA3(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.LISA_arms()
        return xC * l, -lCA, lBC, l, "LISA 3"
    


#***************************************************************************************************************

def detector(det_name: str, shift_angle=None):
    """
    Method to return the coordinates of the observatories
    """
    observatories = Observatories()
    detectors = {
        "LIGO H": observatories.LIGO_hanford,
        "LIGO L": observatories.LIGO_livingston,
        "Virgo": observatories.Virgo,
        "KAGRA": observatories.KAGRA,
        "CE": observatories.CE,
        "ET A": observatories.ET_A,
        "ET B": observatories.ET_B,
        "ET C": observatories.ET_C,
        "ET L1": observatories.ET_L_sardinia,
        "ET L2": lambda angle=shift_angle: observatories.ET_L_netherlands(angle),
        "LISA 1": observatories.LISA1,
        "LISA 2": observatories.LISA2,
        "LISA 3": observatories.LISA3,
    }
    
    if det_name in detectors:
        return detectors[det_name]()
    else:
        print(f"Detector '{det_name}' not found")
        


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#===============================================================
#                      NOISE CURVES
#===============================================================


def detector_Pn(det_name):
    """
    Method to return the PSD of the observatories
    """
    base_path = 'psd/'  # Set a base path for easier management
    file_map = {
        'LIGO H': 'aligo_design.txt',
        'LIGO L': 'aligo_design.txt',
        'Virgo':  'advirgo.txt',
        'KAGRA':  'kagra.txt',
        'ET L1':  '18213_ET15kmcolumns.txt',
        'ET L2':  '18213_ET15kmcolumns.txt',
        'ET A':   'ET_Sh_coba.txt',
        'ET B':   'ET_Sh_coba.txt',
        'ET C':   'ET_Sh_coba.txt',
        'CE':     'ce1.txt',
        'LISA 1': 'lisa_noise.txt', # A channel
        'LISA 2': 'lisa_noise.txt', # A channel
        'LISA 3': 'lisa_noise.txt'  # A channel
    }

    if det_name.startswith('ET L'):
        file_name = file_map[det_name]
        f, Pn = np.loadtxt(base_path + file_name, delimiter= None, usecols=(0,3), unpack=True)
        return f, Pn
    
    elif det_name in file_map:
        file_name = file_map[det_name]
        f, Pn = np.loadtxt(base_path + file_name, unpack=True)
        return f, Pn**2

    else:
        raise ValueError(f'Unknown detector name: {det_name}')


def LISA_noise_AET(f, channel):
        L = 2.5*1e9 #m
        c = 3*1e8 #m/s
        P = 15
        A = 3
        pm = 1e-12 #m
        fm = 1e-15 #m
        
        
        def Poms(f):
            return P*P * pm*pm * ( 1 + (2*1e-3/f)**4 )* (2*np.pi * f /c)**2

        def Pacc(f):
            return A*A * fm*fm * ( 1 + (0.4*1e-3/f)**2 ) * (1 + (f/(8*1e-3))**4 ) * (1/(2* np.pi*f))**4 * (2* np.pi* f /c)**2

        def N_AA(f):
            arg = 2*np.pi*f*L/c
            return 8* np.sin(arg)**2 * (4* (1+ np.cos(arg) + np.cos(arg**2))*Pacc(f) + (2+ np.cos(arg))*Poms(f) )
        
        def N_TT(f):
            arg = 2*np.pi*f*L/c
            return 16* np.sin(arg)**2 * (2*(1- np.cos(arg))**2 * Pacc(f) + (1-np.cos(arg))* Poms(f) )
        
        def R_AA(f):
            arg = 2*np.pi*f*L/c
            return 16* np.sin(arg)**2 * arg**2 * (9/20 /(1 + 0.7*arg**2))
        
        def R_TT(f):
            arg = 2*np.pi*f*L/c
            return 16* np.sin(arg)**2 * arg**2 * (9* arg**6/20 /(1.8e3  + 0.7*arg**8))
        
        if channel == 'A' or channel== 'E':
            return N_AA(f)/R_AA(f)
        elif channel == 'T':
            return N_TT(f)/R_TT(f)
        else:
            print('Channel not found')
            return 0
    
