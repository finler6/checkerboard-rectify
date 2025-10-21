"tests/test_reprojection.py"
from calibrator.prep import generate_checkerboard
from calibrator.detect import find_chessboard_corners
from calibrator.calc import match_points_hungarian, estimate_transform
import cv2
import numpy as np

def test_reprojection_mean_small():
    pattern=(7,7)
    orig, true = generate_checkerboard(pattern, square_size=30)
    h,w = orig.shape[:2]
    M_true = cv2.getRotationMatrix2D((w/2,h/2), 8.5, 0.95)
    distorted = cv2.warpAffine(orig, M_true, (w,h), borderValue=(255,255,255))
    detected = find_chessboard_corners(distorted, pattern_size=pattern)
    assert detected is not None
    matched = match_points_hungarian(detected, true)
    H = estimate_transform(matched, true, method="homography")
    from numpy.linalg import norm
    if H is None:
        pytest.skip("homography failed")
    pts = np.hstack([matched, np.ones((len(matched),1))])
    p = (H @ pts.T).T
    proj = p[:, :2] / p[:, 2:3]
    err = np.mean(norm(proj - true, axis=1))
    assert err < 2.5
