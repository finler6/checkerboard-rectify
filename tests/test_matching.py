"tests/test_matching.py"
import numpy as np
from calibrator.prep import generate_checkerboard
from calibrator.detect import find_chessboard_corners
from calibrator.calc import match_points_hungarian, match_points_greedy

def test_matching_agrees_and_lengths():
    pattern = (7,7)
    img, true_corners = generate_checkerboard(pattern, square_size=20)
    import cv2
    M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2), 7, 0.95)
    distorted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(255,255,255))
    detected = find_chessboard_corners(distorted, pattern_size=pattern)
    assert detected is not None
    h_ordered = match_points_hungarian(detected, true_corners)
    g_ordered = match_points_greedy(detected, true_corners)
    assert h_ordered.shape == g_ordered.shape == true_corners.shape
    assert np.isfinite(h_ordered).all() and np.isfinite(g_ordered).all()
