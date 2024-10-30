import numpy as np
from numpy import pi, sin, cos
from scipy.spatial.transform import Rotation as R

#***************************************************************************************************************

REarth = 6.371 * 1e6 #m
c = 3*1e8 #m/s

def rotate(x, beta):
    return np.array([x[0]*cos(beta) + x[1]*sin(beta), -x[0]*sin(beta) + x[1]*cos(beta), x[2]])

def rot_axis(vector, rotation_angle, rotation_axis):
    rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)
    rotation = R.from_rotvec(rotation_axis * rotation_angle)
    return rotation.apply(vector)

def rot_angle(v1, v2):
    return np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def find_perp(v1, v2):
    return np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def LIGO_hanford():#LIGO Hanford
    ecH = np.array([-0.33827472, -0.60015338,  0.72483525])*REarth 
    e1H = np.array([-0.22389266154, 0.79983062746, 0.55690487831])
    e2H = np.array([-0.91397818574, 0.02609403989, -0.40492342125])
    l = 4e3 #m
    return ecH, e1H, e2H, l, "LIGO Hanford"


def LIGO_livingston():#LIGO Livingston    
    ecL = np.array([-0.01163537, -0.8609929 ,  0.50848387])*REarth
    e1L = np.array([-0.95457412153, -0.14158077340, -0.26218911324])
    e2L = np.array([0.29774156894, -0.48791033647, -0.82054461286])
    l = 4e3 #m
    return ecL, e1L, e2L, l, "LIGO Livingston"

def Virgo():
    ecV     = REarth *np.array([0.71166465, 0.13195706, 0.69001505])
    e1V   = np.array([-0.701, 0.201, 0.684])
    e2V   = np.array([-0.0485, -0.971, 0.236])
    l = 3e3 #m
    return ecV, e1V, e2V, l, "Virgo"

def KAGRA(): # controllare se i valori sono giusti
    ecK     = REarth *np.array([-0.59149285,  0.54570304,  0.59358605])
    e1K   = np.array([-0.390, -0.838, 0.382])
    e2K   = np.array([0.706, -0.00580, 0.709])
    l = 3e3 #m
    return ecK, e1K, e2K, l, "KAGRA"

def ET_L_sardinia():
    beta = pi/2
    #ET_L1_c = np.array([0.7499728 , 0.12438134, 0.64966921])*REarth
    #ET_L1_c = np.array([0.750734, 0.124943, 0.648682])*REarth
    ET_L1_c = np.array([0.7499727960741946, 0.12438133925959684, 0.6496692139794248])*REarth
    ET_L1_1 = np.array([-0.639881, -0.106494, 0.761506])
    ET_L1_2 = rot_axis(ET_L1_1, pi/2, ET_L1_c)
    l = 1.5e4 #m
    return ET_L1_c, ET_L1_1, ET_L1_2, l, "ET L shaped - Sardinia"
    

def ET_L_netherlands():
    sep_angle = np.deg2rad(0)
    ET_L1_c, ET_L1_1, ET_L1_2, l, old_name = ET_L_sardinia()
    #ET_L2_c = np.array([0.627568, 0.062529, 0.776046])*REarth
    ET_L2_c = np.array([0.6296925601906378, 0.06530072687979514, 0.7740950165900375])*REarth
    ET_L2_1_int = rot_axis(ET_L1_1, sep_angle , ET_L1_c)
    ET_L2_2_int = rot_axis(ET_L1_2, sep_angle, ET_L1_c)
    beta = rot_angle(ET_L1_c, ET_L2_c)
    rot_ax = find_perp(ET_L1_c, ET_L2_c)
    ET_L2_1 = rot_axis(ET_L2_1_int, beta, rot_ax)
    ET_L2_2 = rot_axis(ET_L2_2_int, beta, rot_ax)
    return ET_L2_c, ET_L2_1, ET_L2_2, l, "ET L shaped - NLf ="

def ET_L_netherlands_shift(sep_angle):
    ET_L1_c, ET_L1_1, ET_L1_2, l, old_name = ET_L_sardinia()
    ET_L2_c = np.array([0.627568, 0.062529, 0.776046])*REarth
    ET_L2_1_int = rot_axis(ET_L1_1, sep_angle , ET_L1_c)
    ET_L2_2_int = rot_axis(ET_L1_2, sep_angle, ET_L1_c)
    beta = rot_angle(ET_L1_c, ET_L2_c)
    rot_ax = find_perp(ET_L1_c, ET_L2_c)
    ET_L2_1 = np.array(rot_axis(ET_L2_1_int, beta, rot_ax))
    ET_L2_2 = np.array(rot_axis(ET_L2_2_int, beta, rot_ax))
    return ET_L2_c, ET_L2_1, ET_L2_2, l, "ET L shaped - Netherlands"


def CE():
    CE_c, CE_1, CE_2, l, ligo_hanford_name = LIGO_hanford()
    return CE_c, CE_1, CE_2, 4e4, "Cosmic Explorer"

#triangular shaped

# def ET_arms():
#     REarth = 6.371 * 1e6 #m
#     first = np.array([1., 0., 0.])
#     second = np.array([0., 1., 0.])
#     ETCe    = REarth *np.array([0.7499728 , 0.12438134, 0.64966921])
#     ETarm = 1e4 #m

#     ## These are the positions of the 3 detectors
#     ET1     = ETCe +ETarm *(-.5*first -0.28867513*second) 
#     ET2     = ETCe +ETarm *(+.5*first -0.28867513*second)
#     ET3     = ETCe +ETarm *(+0.57735027*second)

#     ## And these are the arms
#     ETarm1  = (+.5*first -0.28867513*second) -(-.5*first -0.28867513*second)
#     ETarm2  = (+0.57735027*second) -(+.5*first -0.28867513*second)
#     ETarm3  = (-.5*first -0.28867513*second) -(+0.57735027*second)

#     ETarm1_ = ETarm1 / np.linalg.norm(ETarm1)
#     ETarm2_ = ETarm2 / np.linalg.norm(ETarm2)
#     ETarm3_ = ETarm3 / np.linalg.norm(ETarm3)
    
#     return ETCe, ET1, ET2, ET3, ETarm1_, ETarm2_, ETarm3_, ETarm

# def ET_A():
#     ETCe, ET1, ET2, ET3, ETarm1_, ETarm2_, ETarm3_, ETarm = ET_arms()
#     return ET1, ETarm1_, ETarm3_, ETarm, "ET A"

# def ET_B():
#     ETCe, ET1, ET2, ET3, ETarm1_, ETarm2_, ETarm3_, ETarm= ET_arms()
#     return ET2, ETarm2_, ETarm1_, ETarm, "ET B"

# def ET_C():
#     ETCe, ET1, ET2, ET3, ETarm1_, ETarm2_, ETarm3_, ETarm = ET_arms()
#     return ET3, ETarm3_, ETarm2_, ETarm, "ET C"



def ET_arms():
    xA = np.array([0.,0.,0.])
    xB = np.array([+1/2., np.sqrt(3)/2, 0])
    xC = np.array([-1/2, np.sqrt(3)/2, 0])
    lBA = xB - xA
    lCA = xC - xA
    lBC = xB - xC
    ETarm = 1e4 #m

    return xA, xB, xC, lBA, lCA, lBC, ETarm


def ET_A():
    xA, xB, xC, lBA, lCA, lBC, ETarm = ET_arms()
    return xA*ETarm, lBA, lCA, ETarm, "ET A"

def ET_B():
    xA, xB, xC, lBA, lCA, lBC, ETarm = ET_arms()
    return xB*ETarm, -lBC, -lBA, ETarm, "ET B"

def ET_C():
    xA, xB, xC, lBA, lCA, lBC, ETarm = ET_arms()
    return xC*ETarm, -lCA, lBC, ETarm, "ET C"

# LISA arms

# def LISA():
#     Rsolarsys = 1.5e11 #m

#     first = np.array([1., 0., 0.])
#     second = np.array([0., 1., 0.])
#     LISACe    = Rsolarsys *np.array([0.7499728 , 0.12438134, 0.64966921])
#     LISAarm = 2.5e9 #m

#     # ### These are the positions of the 3 detectors
#     LISA1_     = LISACe + LISAarm *(-.5*first -0.28867513*second) 
#     LISA2_     = LISACe + LISAarm *(+.5*first -0.28867513*second)
#     LISA3_     = LISACe + LISAarm *(+0.57735027*second)

#     # ### And these are the arms
#     LISAarm1_  = (+.5*first -0.28867513*second) - (-.5*first -0.28867513*second)
#     LISAarm2_  = (+0.57735027*second) -(+.5*first -0.28867513*second)
#     LISAarm3_  = (-.5*first -0.28867513*second) -(+0.57735027*second)

#     LISAarm1_ = LISAarm1_ #/ np.linalg.norm(LISAarm1_)
#     LISAarm2_ = LISAarm2_ #/ np.linalg.norm(LISAarm2_)
#     LISAarm3_ = LISAarm3_ #/ np.linalg.norm(LISAarm3_)

#     return LISA1_, LISA2_, LISA3_, LISAarm1_, LISAarm2_, LISAarm3_, LISAarm

# def LISA1():
#     LISA1, LISA2, LISA3, LISAarm1_, LISAarm2_, LISAarm3_, LISAarm = LISA()
#     return LISA1, LISAarm1_, LISAarm2_, LISAarm, "LISA 1"

# def LISA2():
#     LISA1, LISA2, LISA3, LISAarm1_, LISAarm2_, LISAarm3_, LISAarm = LISA()
#     return LISA2, -LISAarm3_, -LISAarm1_, LISAarm, "LISA 2"

# def LISA3():
#     LISA1, LISA2, LISA3, LISAarm1_, LISAarm2_, LISAarm3_, LISAarm = LISA()
#     return LISA3, -LISAarm2_, LISAarm3_, LISAarm, "LISA 3"


def LISA():

    LISAarm = 2.5e9
    xA = np.array([0.,0.,0.])
    xB = np.array([+1/2., np.sqrt(3)/2, 0])
    xC = np.array([-1/2, np.sqrt(3)/2, 0])
    lBA = (xB - xA)
    lCA = (xC - xA)
    lBC = (xB - xC)

    return xA, xB, xC, lBA, lCA, lBC, LISAarm

def LISA1():
    xA, xB, xC, lBA, lCA, lBC, LISAarm = LISA()
    return xA*LISAarm, lBA, lCA, LISAarm, "LISA 1"

def LISA2():
    xA, xB, xC, lBA, lCA, lBC, LISAarm = LISA()
    return xB*LISAarm, -lBC, -lBA, LISAarm, "LISA 2"

def LISA3():
    xA, xB, xC, lBA, lCA, lBC, LISAarm = LISA()
    return xC*LISAarm, -lCA, lBC, LISAarm, "LISA 3"


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def detector(det_name, shift_angle):
  if(det_name == 'LIGO H' and shift_angle is None):
      return LIGO_hanford()
  elif(det_name == 'LIGO L' and shift_angle is None):
      return LIGO_livingston()
  elif(det_name == 'Virgo' and shift_angle is None):
      return Virgo()
  elif(det_name == 'KAGRA' and shift_angle is None):
      return KAGRA()
  elif(det_name == 'ET L1' and shift_angle is None):
      return ET_L_sardinia()
  elif(det_name == 'ET L2'):
      return ET_L_netherlands_shift(shift_angle)
  elif(det_name == 'ET L2 best' and shift_angle is None):
      return ET_L_netherlands()
  elif(det_name == 'CE' and shift_angle is None):
      return CE()
  elif(det_name == 'ET A' and shift_angle is None):
      return ET_A()
  elif(det_name == 'ET B' and shift_angle is None):
      return ET_B()
  elif(det_name == 'ET C' and shift_angle is None):
      return ET_C()
  elif(det_name == 'LISA 1' and shift_angle is None):
      return LISA1()
  elif(det_name == 'LISA 2' and shift_angle is None):
      return LISA2()  
  elif(det_name == 'LISA 3' and shift_angle is None):
      return LISA3()
  else:
      print('Detector not found')
      return 0, 0, 0, 0, 0
  

    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ Noise curves ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def LISA_noise_XYZ(f, cross, foreground=False):

    L = 2.5*1e9 #m
    c = 3*1e8 #m/s
    P = 15
    A = 3
    pm = 1e-12 #m
    fm = 1e-15 #m
    
    
    def Poms(P, f, pm, c):
        return P*P * pm*pm * ( 1 + (2*1e-3/f)**4 )* (2*np.pi * f /c)**2

    def Pacc(A, fm, c, f):
        return A*A * fm*fm * ( 1 + (0.4*1e-3/f)**2 ) * (1 + (f/(8*1e-3))**4 ) * (1/(2* np.pi*f))**4 * (2* np.pi* f /c)**2

    def Pn_auto(f, P, A, L): #noise autocorrelation
        return 16 * (np.sin(2*np.pi*f*L/c))**2 * ( Poms(P, f, pm, c) + Pacc(A, fm, c, f)*( 3 + np.cos(4*np.pi*f*L/c) )  )

    def Pn_cross(f, P, A, L): #noise autocorrelation
        return -8 * (np.sin(2*np.pi*f*L/c))**2 * np.cos(2*np.pi*f*L/c) *( Poms(P, f, pm, c) + 4*Pacc(A, fm, c, f) )
    
    def R_auto(f, c, L):
        return 16 * (np.sin(2*np.pi*f*L/c))**2 * (3/10) * (2*np.pi*f*L/c)**2 /(1+ 0.6*(2*np.pi*f*L/c)**2)
    
    def R_cross(f, c, L):
        return 16 * (np.sin(2*np.pi*f*L/c))**2 *  (2*np.pi*f*L/c)**2 * (  ( (3/10) * (2*np.pi*f*L/c)**2 /(1+ 0.6*(2*np.pi*f*L/c)**2)) - ( (9/20) * (2*np.pi*f*L/c)**2 /(1+ 0.7*(2*np.pi*f*L/c)**2)) )

    def Sn_foreground(f):
        exp = np.exp(-f**0.138 -221*f* np.sin(521*f) )
        tanh = 1+ np.tanh(1680*(0.00113 -f))
        return 1.80*1e-44* (f**(-7/3)) * exp * tanh
    
    if cross == True:
        def Sn_cross(f, L, c):
            return Pn_cross (f, P, A, L)/ R_cross(f, c, L)
        if foreground == True:
            return Sn_cross(f, L, c) + Sn_foreground(f)
        else:
            return Sn_cross(f, L, c)
    
    else:
        def Sn_auto(f, L, c):
            return Pn_auto (f, P, A, L)/ R_auto(f, c, L)
        if foreground == True:
            return Sn_auto(f, L, c) + Sn_foreground(f)
        else:
            return Sn_auto(f, L, c)
    
    
def LISA_noise_AET(f, channel, foreground=False):
    L = 2.5*1e9 #m
    c = 3*1e8 #m/s
    P = 15
    A = 3
    pm = 1e-12 #m
    fm = 1e-15 #m
    
    
    def Poms(P, f, pm, c):
        return P*P * pm*pm * ( 1 + (2*1e-3/f)**4 )* (2*np.pi * f /c)**2

    def Pacc(A, fm, c, f):
        return A*A * fm*fm * ( 1 + (0.4*1e-3/f)**2 ) * (1 + (f/(8*1e-3))**4 ) * (1/(2* np.pi*f))**4 * (2* np.pi* f /c)**2

    def Pn_auto(f, P, A, L): #noise autocorrelation
        return 16 * (np.sin(2*np.pi*f*L/c))**2 * ( Poms(P, f, pm, c) + Pacc(A, fm, c, f)*( 3 + np.cos(4*np.pi*f*L/c) )  )

    def Pn_cross(f, P, A, L): #noise autocorrelation
        return -8 * (np.sin(2*np.pi*f*L/c))**2 * np.cos(2*np.pi*f*L/c) *( Poms(P, f, pm, c) + 4*Pacc(A, fm, c, f) )
    
    def R_auto(f, c, L):
        return 16 * (np.sin(2*np.pi*f*L/c))**2 * (3/10) * (2*np.pi*f*L/c)**2 /(1+ 0.6*(2*np.pi*f*L/c)**2)
    
    def R_cross(f, c, L):
        return 16 * (np.sin(2*np.pi*f*L/c))**2 *  (2*np.pi*f*L/c)**2 * (  ( (3/10) * (2*np.pi*f*L/c)**2 /(1+ 0.6*(2*np.pi*f*L/c)**2)) - ( (9/20) * (2*np.pi*f*L/c)**2 /(1+ 0.7*(2*np.pi*f*L/c)**2)) )
    
    def P_AA(f, P, A, L):
        return Pn_auto(f, P, A, L) - Pn_cross(f, P, A, L)
    
    def P_TT(f, P, A, L):
        return Pn_auto(f, P, A, L) + 2*Pn_cross(f, P, A, L)
    
    def R_AA(f, c, L):
        return R_auto(f, c, L) - R_cross(f, c, L)
    
    def R_TT(f, c, L):
        return R_auto(f, c, L) + 2*R_cross(f, c, L)

    def Sn_foreground(f):
        exp = np.exp(-f**0.138 -221*f* np.sin(521*f) )
        tanh = 1+ np.tanh(1680*(0.00113 -f))
        return 1.80*1e-44* (f**(-7/3)) * exp * tanh
    
    if foreground==True:
        if channel=='AA' or channel=='EE':
            return P_AA(f, P, A, L)/R_AA(f, c, L) + Sn_foreground(f)
        elif channel=='TT':
            return P_TT(f, P, A, L)/R_TT(f, c, L) + Sn_foreground(f)
        else:
            print('Channel not found')
            return 0
        
    else:
        if channel=='AA' or channel=='EE':
            return P_AA(f, P, A, L)/R_AA(f, c, L)
        elif channel=='TT':
            return P_TT(f, P, A, L)/R_TT(f, c, L)
        else:
            print('Channel not found')
            return 0


def from_Pn_to_Omega(f, Sn):
    H0 = 67/3 * 1e-18 #1/s
    return 4 * np.pi * np.pi * f**3 * Sn / (3* H0**2)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++ PTA ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def PTA_noise(f):
    DT_inverse = 20/(365*24*3600) #Hz
    s = 100 * 1e-9 #s
    return 2* (s**2)/DT_inverse

def Seff(f):
    return PTA_noise(f) * 12 * (np.pi**2) * f**3

def Omega_eff(f):
    H0 = 67/3 * 1e-18 #1/s
    return 4 * np.pi * np.pi * f**3 * Seff(f) / (3* H0**2)

#===============================================================
#                      NOISE CURVES
#===============================================================

f_lisa = np.logspace(-5, -1, 1000)
P_lisa = LISA_noise_AET(f_lisa, channel='AA')

#print lisa noise in a file
np.savetxt('lisa_noise.txt', np.array([f_lisa, np.sqrt(P_lisa)]).T)

def detector_Pn(det_name):
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
        'LISA 1': 'lisa_noise.txt',
        'LISA 2': 'lisa_noise.txt',
        'LISA 3': 'lisa_noise.txt'
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
    
