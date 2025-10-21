"tests/test_prep.py"
import numpy as np
from calibrator.prep import generate_checkerboard

def test_generate_checkerboard_shape():
    img, corners = generate_checkerboard((7,7), square_size=20)
    assert img is not None
    assert img.shape[0] > 0 and img.shape[1] > 0
    assert corners.shape == (7*7, 2)
    assert np.allclose(corners[0], [20, 20])
    assert np.allclose(corners[-1], [7*20, 7*20]) or corners[-1].shape == (2,)
