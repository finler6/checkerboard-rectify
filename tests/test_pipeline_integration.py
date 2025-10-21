"tests/test_pipeline_integration.py"
import numpy as np
from calibrator.prep import generate_checkerboard
from calibrator.detect import find_chessboard_corners
from calibrator.calc import match_points_hungarian, estimate_transform, apply_transform, compute_metrics, center_crop
import cv2

def test_pipeline_end_to_end_homography_center_metrics():
    pattern=(7,7)
    orig, true_corners = generate_checkerboard(pattern, square_size=30)
    h,w = orig.shape[:2]
    M_true = cv2.getRotationMatrix2D((w/2,h/2), 9.3, 0.95)
    distorted = cv2.warpAffine(orig, M_true, (w,h), borderValue=(255,255,255))
    detected = find_chessboard_corners(distorted, pattern_size=pattern)
    assert detected is not None
    matched = match_points_hungarian(detected, true_corners)
    H = estimate_transform(matched, true_corners, method="homography")
    assert H is not None
    corrected = apply_transform(distorted, H, (w,h), borderMode=cv2.BORDER_REFLECT)
    ref_crop = center_crop(orig, 0.12)
    corr_crop = center_crop(corrected, 0.12)
    metrics = compute_metrics(ref_crop, corr_crop)
    assert metrics["ssim"] > 0.8
    assert metrics["psnr"] > 12
