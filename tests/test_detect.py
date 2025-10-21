"tests/test_detect.py"
import numpy as np
from calibrator.prep import generate_checkerboard
from calibrator.detect import find_chessboard_corners

def test_detect_on_synthetic():
    pattern = (7,7)
    img, true_corners = generate_checkerboard(pattern_size=pattern, square_size=20, noise=0.0)
    corners = find_chessboard_corners(img, pattern_size=pattern)
    assert corners is not None, "corners == None — не удалось найти шахматку на синтетике"
    assert corners.shape == true_corners.shape
    assert np.allclose(corners[0], true_corners[0], atol=2.0)
    assert np.allclose(corners[-1], true_corners[-1], atol=2.0)
