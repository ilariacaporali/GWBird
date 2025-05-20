import numpy as np
from numpy import pi, sin, cos, sqrt
from scipy.spatial.transform import Rotation as R
from gwbird.utils import REarth
from gwbird.psd import psd_dir
from gwbird.NANOGrav import NANOGrav_dir
from pint.models import get_model
import glob



#***************************************************************************************************************
#===============================================================
#                      Detectors
#===============================================================



class Rotations:

    '''
    Class to handle rotations in 3D space
    '''

    def rot_axis(vector, angle, axis):
        """
        Rotate a vector around a given axis by a certain angle

        Parameters:
        - vector: array_like (vector to rotate)
        - angle: float (angle of rotation (radians))
        - axis: array_like (axis of rotation (unitary vector))

        Returns: 
        - rotated vector: array_like
        """
        axis = axis / np.linalg.norm(axis)
        cos_theta = cos(angle)
        sin_theta = sin(angle)
        return cos_theta * vector + sin_theta * np.cross(axis, vector) + (1 - cos_theta) * np.dot(axis, vector) * axis

    def rot_angle(vec1, vec2):
        """
        Calculate the angle between two vectors

        Parameters:
        - vec1: array_like (first vector)
        - vec2: array_like (second vector)

        Returns:
        - angle: float (angle between the two vectors)
        """
        return np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

    def find_perp(vec1, vec2):
        """
        Find a vector perpendicular to two given vectors

        Parameters:
        - vec1: array_like (first vector)
        - vec2: array_like (second vector)

        Returns:
        - perpendicular vector: array_like
        """
        return np.cross(vec1, vec2)



class Observatories:

    '''
    Class to handle the coordinates of the observatories (detectors, both ground and space based)
    '''

    # 2G detectors

    def LIGO_hanford(self): # Mentasti et al. 2023 https://arxiv.org/pdf/2304.06640 Appendix D, figure 8
        c = np.array([-0.33827472, -0.60015338, 0.72483525]) * REarth
        xA = np.array([-0.22389266154, 0.79983062746, 0.55690487831])
        xB = np.array([-0.91397818574, 0.02609403989, -0.40492342125])
        l = 4e3  # m
        return c, xA, xB, l, "LIGO Hanford"

    def LIGO_livingston(self): # Mentasti et al. 2023 https://arxiv.org/pdf/2304.06640 Appendix D, figure 8
        c = np.array([-0.01163537, -0.8609929, 0.50848387]) * REarth
        xA = np.array([-0.95457412153, -0.14158077340, -0.26218911324])
        xB = np.array([0.29774156894, -0.48791033647, -0.82054461286])
        l = 4e3  # m
        return c, xA, xB, l, "LIGO Livingston"

    def Virgo(self): # Mentasti et al. 2023 https://arxiv.org/pdf/2304.06640 Appendix D, figure 8
        c = np.array([0.71166465, 0.13195706, 0.69001505]) * REarth
        xA = np.array([-0.701, 0.201, 0.684])
        xB = np.array([-0.0485, -0.971, 0.236])
        l = 3e3  # m
        return c, xA, xB, l, "Virgo"

    def KAGRA(self): # Mentasti et al. 2023 https://arxiv.org/pdf/2304.06640 Appendix D, figure 8
        c = np.array([-0.59149285, 0.54570304, 0.59358605]) * REarth
        xA = np.array([-0.390, -0.838, 0.382])
        xB = np.array([0.706, -0.00580, 0.709])
        l = 3e3  # m
        return c, xA, xB, l, "KAGRA"
    
    # 3G detectors
    
    # Cosmic Explorer in the location of LIGO Hanford

    def CE(self): # Mentasti et al. 2023 https://arxiv.org/pdf/2304.06640 Appendix D, figure 8
        c, xA, xB, l, _ = self.LIGO_hanford()
        return c, xA, xB, 4e4, "Cosmic Explorer"

    # Einstein Telescope (ET) - triangular configuration

    def ET_arms(self):
        l = 1e4  # m
        c = np.array([0.751, 0.125, 0.649])* REarth

        d1 = np.array([-0.640, -0.106, 0.761])
        d2 = np.array([0.178, 0.908, -0.381])
        d3 = np.array([0.462, -0.801, -0.381])
        
        d = l/2 /cos(np.pi/6)

        v1 = c + d1 * d
        v2 = c + d2 * d
        v3 = c + d3 * d

        arm1 = v2 - v1
        arm2 = v3 - v2
        arm3 = v1 - v3

        arm1 = arm1 / np.linalg.norm(arm1)
        arm2 = arm2 / np.linalg.norm(arm2)
        arm3 = arm3 / np.linalg.norm(arm3)

        return v1, v2, v3, arm1, arm2, arm3, l

    def ET_X(self):
        v1, v2, v3, arm1, arm2, arm3, l = self.ET_arms()
        xA = arm1
        xB = -arm3
        return v1, xA, xB, l, "ET X"
    
    def ET_Y(self):
        v1, v2, v3, arm1, arm2, arm3, l = self.ET_arms()
        xA = arm2
        xB = -arm1
        return v2, xA, xB, l, "ET Y"
    
    def ET_Z(self):
        v1, v2, v3, arm1, arm2, arm3, l = self.ET_arms()
        xA = arm3
        xB = -arm2
        return v3, xA, xB, l, "ET Z"

    
    # Einstein Telescope (ET) - L-shaped configuration

    def ET_L_sardinia(self): # Branchesi et al 2023 https://arxiv.org/pdf/2303.15923
        c = np.array([0.7499728, 0.12438134, 0.64966921]) * REarth
        xA = np.array([-0.639881, -0.106494, 0.761506])
        xB = Rotations.rot_axis(xA, pi/2, c)
        l = 1.5e4 # m
        return c, xA, xB, l, "ET L-shaped Sardinia"

    def ET_L_netherlands(self, shift_angle=None): # Branchesi et al 2023 https://arxiv.org/pdf/2303.15923
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
    return ['LIGO H', 'LIGO L', 'Virgo', 'KAGRA', 'CE', 'ET X', 'ET Y', 'ET Z', 'ET A', 'ET E', 'ET T', 'ET L1', 'ET L2', 'LISA X', 'LISA Y', 'LISA Z', 'LISA A', 'LISA E', 'LISA T']
    

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#===============================================================
#                      NOISE CURVES (for detectors)
#===============================================================


def detector_Pn(det_name):
    """
    Method to return the PSD of the observatories
    """
    base_path = psd_dir  # path to the PSD files # aggiungere referenze
    file_map = {
        'LIGO H': 'aligo_design.txt', # https://dcc.ligo.org/LIGO-T1500293/public
        'LIGO L': 'aligo_design.txt', # https://dcc.ligo.org/LIGO-T1500293/public
        'Virgo':  'advirgo.txt', # https://dcc.ligo.org/LIGO-T1500293/public
        'KAGRA':  'kagra.txt', # https://dcc.ligo.org/LIGO-T1500293/public
        'ET L1':  '18213_ET15kmcolumns.txt', # Coba
        'ET L2':  '18213_ET15kmcolumns.txt', # Coba
        'ET X':   'ET_Sh_coba.txt', # Coba
        'ET Y':   'ET_Sh_coba.txt', # Coba
        'ET Z':   'ET_Sh_coba.txt', # Coba
        'CE':     'ce1.txt', # https://dcc.ligo.org/LIGO-T1500293/public
        'LISA X': 'lisa_noise.txt', # A channel 
        'LISA Y': 'lisa_noise.txt', # A channel
        'LISA Z': 'lisa_noise.txt'  # A channel
    }

    if det_name.startswith('ET L'):
        file_name = file_map[det_name]
        f, Pn = np.loadtxt(base_path + '/' + file_name, delimiter= None, usecols=(0,3), unpack=True)
        return f, Pn
    
    elif det_name in file_map:
        file_name = file_map[det_name]
        f, Pn = np.loadtxt(base_path + '/' + file_name, unpack=True)
        return f, Pn**2

    else:
        raise ValueError(f'Unknown detector name: {det_name}')


def LISA_noise_AET(f, channel):

    '''
    Method to return the noise curve for the LISA detector in the A, E and T channels

    Parameters:
    - f: array_like (frequency)
    - channel: str (channel name)

    Returns:
    - noise curve: array_like
    '''
        
    
    # Bartolo et al. 2022 https://arxiv.org/abs/2201.08782 Appendix B
    
    L = 2.5*1e9 #m
    c = 3*1e8 #m/s
    P = 15
    A = 3
    pm = 1e-12 #m
    fm = 1e-15 #m
    f_star =c/(2*np.pi*L) # Hz


    def NA(f): #  Bartolo et al. 2022 https://arxiv.org/abs/2201.08782 Appendix B eq. B.14
        N = 0.5 * (2 + cos(f/f_star)) * P**2 / L**2 * pm**2 * ( 1 + (0.002/f)**4 ) + \
            2 * ( 1 + cos(f/f_star) + cos(f/f_star)**2 ) * A**2 / L**2 *fm**2 * (1 + (0.0004/f)**2 ) * (1 + (f/(8*1e-3))**4 ) * ((1/(2* pi*f))**4)
        return N

    def NT(f): #  Bartolo et al. 2022 https://arxiv.org/abs/2201.08782 Appendix B eq. B.15
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
        
#***************************************************************************************************************
#===============================================================
#                      Pulsar Timing Arrays
#===============================================================

def get_NANOGrav_pulsars(): # https://zenodo.org/records/14773896

    '''
    Function to get the pulsar data from the NANOGrav dataset

    Returns:
    - N_pulsar: int (number of pulsars)
    - pulsar_xyz: array_like (pulsar coordinates)
    - DIST_array: array_like (pulsar distances) # in meters

    '''

    pfiles = glob.glob(NANOGrav_dir+ '/'+'*.par')
    pfiles = [pf for pf in pfiles if not 'gbt' in pf and not 'ao' in pf]
    pnames = [pf[4:pf.index('_PINT')] for pf in pfiles]

    RA = {}   # Right Ascension
    DEC = {}  # Declination
    DIST = {} # Distance (in parsec)

    for pf in pfiles:
        m = get_model(pf)
        c = m.components['AstrometryEcliptic'].coords_as_ICRS()
        RA[m.PSR.value] = c.ra.deg  
        DEC[m.PSR.value] = c.dec.deg

        if hasattr(m, 'PX') and m.PX.value > 0:  
            DIST[m.PSR.value] = 1000 / m.PX.value  
        else:
            DIST[m.PSR.value] = None  

    # parsec 2 meters
    for psr in DIST.keys():
        if DIST[psr] is not None:
            DIST[psr] *= 3.086e16

    # ra dec 2  theta phi 

    theta_pulsar = np.deg2rad(90.0 - np.array(list(DEC.values())))
    phi_pulsar = np.deg2rad(list(RA.values()))

    # dictionaries 2 arrays
    valid_indices = [psr for psr in DIST if DIST[psr] is not None] 
    DIST_array = np.array([DIST[psr] for psr in valid_indices])
    theta_pulsar = np.deg2rad(90.0 - np.array([DEC[psr] for psr in valid_indices]))
    phi_pulsar = np.deg2rad([RA[psr] for psr in valid_indices])

    # spherical 2 cartesian
    x_pulsar = np.sin(theta_pulsar) * np.cos(phi_pulsar)
    y_pulsar = np.sin(theta_pulsar) * np.sin(phi_pulsar)
    z_pulsar = np.cos(theta_pulsar)
    pulsar_xyz = np.array([x_pulsar, y_pulsar, z_pulsar]).T

    N_pulsar = len(valid_indices)

    return N_pulsar, pulsar_xyz, DIST_array

