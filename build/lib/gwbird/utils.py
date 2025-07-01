from astropy.cosmology import Planck18

cosmo = Planck18
H0 =  cosmo.H0.to('1/s').value
h = 0.7
c = 299792458 # speed of light
REarth = 6.371 * 1e6 #m