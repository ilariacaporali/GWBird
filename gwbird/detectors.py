import numpy as np
from numpy import pi, sin, cos, sqrt
from scipy.spatial.transform import Rotation as R
from gwbird.utils import REarth


#***************************************************************************************************************



class Rotations:


    def rot_axis(vector, angle, axis):
        """
        Rotate a vector around a given axis by a certain angle
        """
        axis = axis / np.linalg.norm(axis)
        cos_theta = cos(angle)
        sin_theta = sin(angle)
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

    def LIGO_hanford(self): # aggiungere referenza
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
    
    # 3G detectors
    
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

    def ET_X(self):
        xA, xB, xC, lBA, lCA, lBC, l= self.ET_arms()
        return xA * l, lBA, lCA, l, "ET X"

    def ET_Y(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.ET_arms()
        return xB * l, -lBC, -lBA, l, "ET Y"

    def ET_Z(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.ET_arms()
        return xC * l, -lCA, lBC, l, "ET Z"
    
    # Einstein Telescope (ET) - L-shaped configuration

    def ET_L_sardinia(self):
        c = np.array([0.7499728, 0.12438134, 0.64966921]) * REarth
        xA = np.array([-0.639881, -0.106494, 0.761506])
        xB = Rotations.rot_axis(xA, pi/2, c)
        l = 1.5e4 # m
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

    # space-based

    # mettere X, Y, Z

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

    def LISA_X(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.LISA_arms()
        return xA * l, lBA, lCA, l, "LISA X"

    def LISA_Y(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.LISA_arms()
        return xB * l, -lBC, -lBA, l, "LISA Y"

    def LISA_Z(self):
        xA, xB, xC, lBA, lCA, lBC, l = self.LISA_arms()
        return xC * l, -lCA, lBC, l, "LISA Z"
    


#***************************************************************************************************************

def detector(det_name: str, shift_angle=None, c=None, xA=None, xB=None, l=None, name=None):
    """
    Method to return the coordinates of the observatories
    """
    observatories = Observatories()
    detectors = {
        "LIGO H": observatories.LIGO_hanford,
        "LIGO L": observatories.LIGO_livingston,
        "Virgo":  observatories.Virgo,
        "KAGRA":  observatories.KAGRA,
        "CE":     observatories.CE,
        "ET X":   observatories.ET_X, 
        "ET Y":   observatories.ET_Y,
        "ET Z":   observatories.ET_Z,
        "ET L1":  observatories.ET_L_sardinia,
        "ET L2":  lambda angle=shift_angle: observatories.ET_L_netherlands(angle),
        "LISA X": observatories.LISA_X, 
        "LISA Y": observatories.LISA_Y,
        "LISA Z": observatories.LISA_Z
    }
    
    if det_name in detectors:
        return detectors[det_name]()
    
    elif isinstance(c, np.ndarray) and isinstance(xA, np.ndarray) and isinstance(xB, np.ndarray) and isinstance(l, (int, float)) and isinstance(name, str):
        return c, xA, xB, l, name

    else:
        print(f"Detector '{det_name}' not found")
        

def available_detectors():
    '''
    List of available detectors
    '''
    return ['LIGO H', 'LIGO L', 'Virgo', 'KAGRA', 'CE', 'ET X', 'ET Y', 'ET Z', 'ET L1', 'ET L2', 'LISA X', 'LISA Y', 'LISA Z']
    

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#===============================================================
#                      NOISE CURVES
#===============================================================


def detector_Pn(det_name):
    """
    Method to return the PSD of the observatories
    """
    base_path = 'psd/'  # path to the PSD files # aggiungere referenze
    file_map = {
        'LIGO H': 'aligo_design.txt',
        'LIGO L': 'aligo_design.txt',
        'Virgo':  'advirgo.txt',
        'KAGRA':  'kagra.txt',
        'ET L1':  '18213_ET15kmcolumns.txt',
        'ET L2':  '18213_ET15kmcolumns.txt',
        'ET X':   'ET_Sh_coba.txt',
        'ET Y':   'ET_Sh_coba.txt',
        'ET Z':   'ET_Sh_coba.txt',
        'CE':     'ce1.txt',
        'LISA X': 'lisa_noise.txt', # A channel # modificare
        'LISA Y': 'lisa_noise.txt', # A channel
        'LISA Z': 'lisa_noise.txt'  # A channel
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
        f_star =c/(2*np.pi*L) # Hz


        def NA(f):
            N = 0.5 * (2 + cos(f/f_star)) * P**2 / L**2 * pm**2 * ( 1 + (0.002/f)**4 ) + \
                2 * ( 1 + cos(f/f_star) + cos(f/f_star)**2 ) * A**2 / L**2 *fm**2 * (1 + (0.0004/f)**2 ) * (1 + (f/(8*1e-3))**4 ) * ((1/(2* pi*f))**4)
            return N
    
        def NT(f):
            N = (1 - cos(f/f_star)) * P**2 / L**2 * pm**2 * ( 1 + (0.002/f)**4 ) + \
                2 * ( 1 - cos(f/f_star) )**2 * A**2 / L**2 *fm**2 * (1 + (0.0004/f)**2 ) * (1 + (f/(0.008))**4 ) * ((1/(2* pi*f))**4)
            return N
            
        if channel == 'A':
            return NA(f)
        elif channel == 'E':
            return NA(f)
        elif channel == 'T':
            return NT(f)
        else:
            raise ValueError('Unknown channel')