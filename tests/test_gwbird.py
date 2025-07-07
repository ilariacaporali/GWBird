import pytest
import numpy as np
import gwbird
from gwbird.skymap import Basis, Skymaps
from gwbird.overlap import Response
from gwbird import pls

def test_skymaps():
    mapHL = Skymaps.antennapattern('LIGO H', 'LIGO L', 10, 0, 't')

def test_Response():
    f = np.logspace(0, 3, 100)
    R_ET2L45 = Response.overlap('ET L1', 'ET L2', f, 't', 0, shift_angle=np.pi/4)

def test_Response_PTA():
    f = np.logspace(-9, -7, 100)
    gamma_NANOGrav = Response.overlap_PTA(f, 't', 0)

def test_pls():
    f = np.logspace(0, 3, 100)
    fref = 10
    snr=1
    Tobs=1
    psi=0
    sens_LIGO = pls.PLS('LIGO H', 'LIGO L', f, fref, 't', snr, Tobs, psi)

def test_pls_PTA():
    f = np.logspace(-9, -7, 100)
    snr=1
    Tobs=15
    psi=0
    pls_PTA = pls.PLS_PTA(f, snr, Tobs, 't', psi)

if __name__ == "__main__":
    pytest.main([__file__])