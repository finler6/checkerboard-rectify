"tests/test_refine.py"
import numpy as np
import cv2
import pytest
from calibrator.prep import generate_checkerboard
from calibrator.detect import find_chessboard_corners
from calibrator.calc import match_points_hungarian, refine_homography_leastsq

def test_refine_reduces_error():
    pattern=(7,7)
    orig, true = generate_checkerboard(pattern, square_size=30)
    h,w = orig.shape[:2]
    Mtrue = cv2.getRotationMatrix2D((w/2,h/2), 12.3, 0.92)
    distorted = cv2.warpAffine(orig, Mtrue, (w,h), borderValue=(255,255,255))
    detected = find_chessboard_corners(distorted, pattern)
    assert detected is not None
    matched = match_points_hungarian(detected, true)
    H_init, mask = cv2.findHomography(matched, true, cv2.RANSAC, 3.0)
    inliers = mask.ravel().nonzero()[0]
    if len(inliers) < 6:
        pytest.skip("too few inliers")
    def reproj(H, src, dst):
        src_h = np.hstack([src, np.ones((len(src),1))])
        p = (H @ src_h.T).T
        p2 = p[:,:2] / p[:,2:3]
        return np.mean(np.linalg.norm(p2 - dst, axis=1))
    err_init = reproj(H_init, matched[inliers], true[inliers])
    H_ref, info = refine_homography_leastsq(H_init, matched[inliers], true[inliers])
    err_ref = reproj(H_ref, matched[inliers], true[inliers])
    assert err_ref <= err_init + 1e-6
