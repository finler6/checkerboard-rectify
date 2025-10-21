"tests/test_model_select.py"
import numpy as np
import cv2
from calibrator.prep import generate_checkerboard
from calibrator.detect import find_chessboard_corners
from calibrator.calc import match_points_hungarian, choose_best_transform, reproj_rms

def make_affine(center, angle=10.0, scale=0.93, tx=15, ty=-8):
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0,2] += tx; M[1,2] += ty
    return M

def test_choose_affine_on_affine():
    img, true_corners = generate_checkerboard((7,7), 40, noise=0.0)
    h, w = img.shape[:2]
    M = make_affine((w/2,h/2))
    warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    det = find_chessboard_corners(warped, (7,7))
    assert det is not None
    det_m = match_points_hungarian(det, true_corners)
    M0, method, inl = choose_best_transform(det_m, true_corners)
    assert method == "affine"
    assert len(inl) >= 10  # должна быть адекватная доля инлаеров
    assert reproj_rms(M0, det_m, true_corners) < 3.0

def test_choose_homography_on_perspective():
    img, true_corners = generate_checkerboard((7,7), 40, noise=0.0)
    h, w = img.shape[:2]
    src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dst = np.float32([[10,20],[w-20,5],[w-5,h-25],[15,h-10]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    det = find_chessboard_corners(warped, (7,7))
    assert det is not None
    det_m = match_points_hungarian(det, true_corners)
    M0, method, inl = choose_best_transform(det_m, true_corners)
    assert method == "homography"
    assert len(inl) >= 10
