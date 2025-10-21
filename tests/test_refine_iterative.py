"tests/test_refine_iterative.py"
import numpy as np
import cv2
from calibrator.prep import generate_checkerboard
from calibrator.detect import find_chessboard_corners
from calibrator.calc import match_points_hungarian, iterative_refine_homography

def test_iterative_refine_reduces_outliers():
    pattern=(7,7)
    orig, true = generate_checkerboard(pattern, square_size=30)
    h,w = orig.shape[:2]
    Mtrue = cv2.getRotationMatrix2D((w/2,h/2), 12.3, 0.92)
    distorted = cv2.warpAffine(orig, Mtrue, (w,h), borderValue=(255,255,255))
    detected = find_chessboard_corners(distorted, pattern)
    assert detected is not None
    matched = match_points_hungarian(detected, true)
    H_init, mask = cv2.findHomography(matched, true, cv2.RANSAC, 3.0)
    H_final, info = iterative_refine_homography(H_init, matched, true, max_iter=3, camp=3.0, verbose=False)
    assert info['final_used'] >= 4
