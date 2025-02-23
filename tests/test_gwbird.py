import pytest
import numpy as np

def test_gwbird():
    import gwbird

def test_Response():
    from gwbird.overlap import Response
    f = np.logspace(0, 3, 100)
    R_ET2L45 = Response.overlap('ET L1', 'ET L2', f, 0, 't', shift_angle=np.pi/4)
    
if __name__ == "__main__":
    pytest.main([__file__])