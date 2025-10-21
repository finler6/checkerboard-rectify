"calibrator/calc.py"

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from typing import Optional, Tuple, Dict, Any


def estimate_transform(src_pts: np.ndarray, dst_pts: np.ndarray,
                       method: str = "affine") -> Optional[np.ndarray]:
    if src_pts is None or dst_pts is None:
        return None
    if len(src_pts) < 3 or len(dst_pts) < 3:
        return None
    src = src_pts.reshape(-1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 2).astype(np.float32)
    if method == "homography":
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        return H
    else:
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        return M

def apply_transform(img: np.ndarray, M: np.ndarray, dsize: Tuple[int, int],
                    borderMode=cv2.BORDER_CONSTANT) -> np.ndarray:
    if M is None:
        raise ValueError("Transformation matrix M is None")
    if M.shape == (2, 3):
        return cv2.warpAffine(img, M, dsize, flags=cv2.INTER_LINEAR,
                              borderMode=borderMode, borderValue=(255,255,255))
    elif M.shape == (3, 3):
        return cv2.warpPerspective(img, M, dsize, flags=cv2.INTER_LINEAR,
                                   borderMode=borderMode, borderValue=(255,255,255))
    else:
        raise ValueError(f"Unsupported transform shape: {M.shape}")

def _to_gray_uint8(im: np.ndarray) -> np.ndarray:
    if im.ndim == 3 and im.shape[2] == 3:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        g = im.copy()
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g

def compute_metrics(img_ref: np.ndarray, img_test: np.ndarray) -> Dict[str, Any]:
    if img_ref.shape != img_test.shape:
        raise ValueError("Images must have same shape for metrics")
    ref_gray = _to_gray_uint8(img_ref)
    test_gray = _to_gray_uint8(img_test)
    diff = ref_gray.astype(np.float32) - test_gray.astype(np.float32)
    mse = float(np.mean(diff**2))
    try:
        psnr_val = float(cv2.PSNR(ref_gray, test_gray))
    except Exception:
        if mse == 0:
            psnr_val = float("inf")
        else:
            PIXEL_MAX = 255.0
            psnr_val = 20 * float(np.log10(PIXEL_MAX / np.sqrt(mse)))
    ssim_val = float(ssim(ref_gray, test_gray, data_range=255))
    return {"mse": mse, "psnr": psnr_val, "ssim": ssim_val}

def compute_metrics_binary(ref_bin: np.ndarray, test_bin: np.ndarray) -> Dict[str, Any]:
    a = (ref_bin > 127).astype(np.uint8)
    b = (test_bin > 127).astype(np.uint8)
    inter = int(np.logical_and(a == 1, b == 1).sum())
    union = int(np.logical_or(a == 1, b == 1).sum())
    iou = float(inter / union) if union > 0 else 1.0
    acc = float((a == b).mean())
    return {"acc": acc, "iou": iou}

def compute_metrics_binary_masked(ref_bin: np.ndarray, test_bin: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    m = (mask > 0)
    if m.ndim == 3:
        m = m[..., 0] > 0
    a = (ref_bin > 127)
    b = (test_bin > 127)
    a = a[m]
    b = b[m]
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    iou = float(inter / union) if union > 0 else 1.0
    acc = float((a == b).mean()) if a.size > 0 else 1.0
    return {"acc": acc, "iou": iou}


def _linear_photometric_align(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = (mask > 0)
    y = ref[m].astype(np.float64).ravel()
    x = test[m].astype(np.float64).ravel()
    if x.size < 10:
        return test
    X = np.stack([x, np.ones_like(x)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeff
    out = a * test.astype(np.float64) + b
    return np.clip(out, 0, 255).astype(np.uint8)

def compute_metrics_masked(img_ref: np.ndarray, img_test: np.ndarray, mask: np.ndarray,
                           photometric: bool = True, blur_sigma: float = 0.0,
                           downscale_factor: float = 1.0,
                           photometric_method: str = "linear") -> Dict[str, Any]:
    if img_ref.shape[:2] != img_test.shape[:2]:
        raise ValueError("Images must have same HxW for masked metrics")
    if mask.shape[:2] != img_ref.shape[:2]:
        raise ValueError("Mask must match image size")
    ref_gray = _to_gray_uint8(img_ref)
    test_gray = _to_gray_uint8(img_test)
    if photometric:
        if photometric_method == "histogram":
            m = (mask > 0)
            ref_vals = ref_gray[m]
            tst_vals = test_gray[m]
            if ref_vals.size > 0 and tst_vals.size > 0:
                hist_ref, bins = np.histogram(ref_vals, bins=256, range=(0,255), density=True)
                cdf_ref = np.cumsum(hist_ref)
                cdf_ref = (cdf_ref / cdf_ref[-1])
                hist_tst, _ = np.histogram(tst_vals, bins=256, range=(0,255), density=True)
                cdf_tst = np.cumsum(hist_tst)
                cdf_tst = (cdf_tst / cdf_tst[-1])
                lut = np.interp(test_gray.ravel(), np.arange(256), np.interp(cdf_tst, cdf_ref, np.arange(256)))
                test_gray = lut.reshape(test_gray.shape).astype(np.uint8)
        else:
            test_gray = _linear_photometric_align(ref_gray, test_gray, mask)

    if isinstance(blur_sigma, (int, float)) and blur_sigma > 0:
        k = int(max(3, round(blur_sigma * 6)))
        if k % 2 == 0:
            k += 1
        ref_gray = cv2.GaussianBlur(ref_gray, (k, k), blur_sigma)
        test_gray = cv2.GaussianBlur(test_gray, (k, k), blur_sigma)

    if isinstance(downscale_factor, (int, float)) and 0.2 <= downscale_factor < 1.0:
        H, W = ref_gray.shape[:2]
        Wd = max(8, int(W * downscale_factor))
        Hd = max(8, int(H * downscale_factor))
        ref_gray = cv2.resize(ref_gray, (Wd, Hd), interpolation=cv2.INTER_AREA)
        test_gray = cv2.resize(test_gray, (Wd, Hd), interpolation=cv2.INTER_AREA)
        mask = cv2.resize((mask>0).astype(np.uint8), (Wd, Hd), interpolation=cv2.INTER_NEAREST)

    m = (mask > 0).astype(np.float32)
    diff = (ref_gray.astype(np.float32) - test_gray.astype(np.float32))**2
    denom = float(m.sum()) if m.sum() > 0 else 1.0
    mse = float((diff * m).sum() / denom)
    if mse == 0:
        psnr_val = float("inf")
    else:
        PIXEL_MAX = 255.0
        psnr_val = 20 * float(np.log10(PIXEL_MAX / np.sqrt(mse)))
    ssim_val, ssim_map = ssim(ref_gray, test_gray, data_range=255, full=True)
    ssim_masked = float((ssim_map * m).sum() / denom)
    return {"mse": mse, "psnr": psnr_val, "ssim": ssim_masked}

def center_crop(img: np.ndarray, crop_ratio: float=0.1) -> np.ndarray:
    h, w = img.shape[:2]
    m = int(min(h, w) * crop_ratio)
    if m == 0:
        return img.copy()
    return img[m:-m, m:-m]


def match_points_greedy(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if len(src) != len(dst):
        raise ValueError("src_pts and dst_pts must have same length for greedy matching")
    used = np.zeros(len(src), dtype=bool)
    assigned_indices = np.empty(len(dst), dtype=int)
    for i in range(len(dst)):
        dists = np.linalg.norm(src - dst[i], axis=1)
        dists[used] = np.inf
        idx = int(np.argmin(dists))
        assigned_indices[i] = idx
        used[idx] = True
    matched_src = src[assigned_indices]
    return matched_src

def match_points_hungarian(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Global optimal assignment (min sum distances) via Hungarian algorithm."""
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:
        raise RuntimeError("scipy is required for Hungarian matching. Install scipy.") from e
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if len(src) != len(dst):
        raise ValueError("src_pts and dst_pts must have same length for Hungarian matching")
    D = np.linalg.norm(dst[:, None, :] - src[None, :, :], axis=2)  # (Ndst, Nsrc)
    row_ind, col_ind = linear_sum_assignment(D)
    ordered_src = np.empty_like(src)
    ordered_src[row_ind] = src[col_ind]
    return ordered_src


def board_mask_from_corners(image_shape: Tuple[int,int], inner_corners: np.ndarray,
                            grow_px: Optional[int]=None) -> np.ndarray:
    H, W = image_shape[:2] if isinstance(image_shape, tuple) else image_shape
    pts = np.asarray(inner_corners, dtype=np.float32).reshape(-1,2)
    if len(pts) < 4:
        return np.zeros((H,W), np.uint8)
    hull = cv2.convexHull(pts).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if grow_px is None:
        d = np.sqrt(((pts[None,:,:] - pts[:,None,:])**2).sum(-1))
        np.fill_diagonal(d, np.inf)
        med_nn = float(np.median(d.min(axis=1))) if np.isfinite(d).any() else 4.0
        grow_px = max(2, int(0.5 * med_nn))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*grow_px+1, 2*grow_px+1))
    mask = cv2.dilate(mask, k, iterations=1)
    return mask


def _h_from_params(params: np.ndarray) -> np.ndarray:
    h = np.array([params[0], params[1], params[2],
                  params[3], params[4], params[5],
                  params[6], params[7], 1.0], dtype=np.float64)
    return h.reshape(3,3)

def _proj_points_homography(h: np.ndarray, src: np.ndarray) -> np.ndarray:
    src_h = np.hstack([src, np.ones((src.shape[0],1), dtype=np.float64)])
    p = (h @ src_h.T).T
    return (p[:, :2] / p[:, 2:3])

def refine_homography_leastsq(H_init: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
                              max_nfev: int = 2000, verbose: bool=False) -> tuple:
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        raise RuntimeError("scipy is required for homography refinement. Install scipy.") from e
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 2
    H0 = H_init.astype(np.float64)
    if abs(H0[2,2]) < 1e-12:
        H0[2,2] = 1.0
    H0 = H0 / H0[2,2]
    p0 = np.array([H0[0,0], H0[0,1], H0[0,2],
                   H0[1,0], H0[1,1], H0[1,2],
                   H0[2,0], H0[2,1]], dtype=np.float64)
    def _h_from_params_local(params):
        h = np.array([params[0], params[1], params[2],
                      params[3], params[4], params[5],
                      params[6], params[7], 1.0], dtype=np.float64)
        return h.reshape(3,3)
    def residuals(p):
        h = _h_from_params_local(p)
        src_h = np.hstack([src, np.ones((src.shape[0],1), dtype=np.float64)])
        proj = (h @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return (proj - dst).ravel()
    res = least_squares(residuals, p0, method='lm', max_nfev=max_nfev, xtol=1e-12, ftol=1e-12)
    p_opt = res.x
    H_ref = _h_from_params_local(p_opt)
    src_h = np.hstack([src, np.ones((src.shape[0],1), dtype=np.float64)])
    p = (H_ref @ src_h.T).T
    proj = p[:, :2] / p[:, 2:3]
    d = np.linalg.norm(proj - dst, axis=1)
    info = {"mean_err": float(d.mean()), "std_err": float(d.std()), "max_err": float(d.max()),
            "success": bool(res.success)}
    if verbose:
        print("LM refine:", info, "cost:", res.cost, "nfev:", res.nfev)
    return H_ref, info


def reproj_rms(M: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if M is None:
        return np.inf
    if M.shape == (2,3):
        src_h = np.hstack([src, np.ones((len(src),1))])
        proj = (M @ src_h.T).T
    elif M.shape == (3,3):
        src_h = np.hstack([src, np.ones((len(src),1))])
        p = (M @ src_h.T).T
        proj = p[:, :2] / p[:, 2:3]
    else:
        return np.inf
    return float(np.sqrt(np.mean(np.sum((proj - dst)**2, axis=1))))

def _fit_affine_ls(src_in: np.ndarray, dst_in: np.ndarray) -> Optional[np.ndarray]:
    src = np.asarray(src_in, dtype=np.float64)
    dst = np.asarray(dst_in, dtype=np.float64)
    if len(src) < 3:
        return None
    A = np.zeros((2*len(src), 6), dtype=np.float64)
    b = np.zeros((2*len(src),), dtype=np.float64)
    A[0::2, 0] = src[:, 0]; A[0::2, 1] = src[:, 1]; A[0::2, 2] = 1.0
    A[1::2, 3] = src[:, 0]; A[1::2, 4] = src[:, 1]; A[1::2, 5] = 1.0
    b[0::2] = dst[:, 0]; b[1::2] = dst[:, 1]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    M = np.array([[x[0], x[1], x[2]],[x[3], x[4], x[5]]], dtype=np.float64)
    return M

def choose_best_transform(src_pts: np.ndarray, dst_pts: np.ndarray, ransac_thresh: float = 3.0):
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    H, maskH = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    inH = np.where(maskH.ravel() != 0)[0] if maskH is not None else np.arange(len(src))
    errH = reproj_rms(H, src, dst)
    A, maskA = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    inA = np.where(maskA.ravel() != 0)[0] if maskA is not None else np.arange(len(src))
    A_ref = _fit_affine_ls(src[inA], dst[inA]) if len(inA) >= 3 else A
    errA = reproj_rms(A_ref, src, dst)
    if errA <= errH * 0.98:
        return A_ref, "affine", inA
    else:
        return H, "homography", inH


def reorder_chessboard_to_reference(detected_pts: np.ndarray, pattern_size: Tuple[int,int],
                                    reference_pts: np.ndarray) -> np.ndarray:
    det = np.asarray(detected_pts, np.float32).reshape(-1,2)
    ref = np.asarray(reference_pts, np.float32).reshape(-1,2)
    C, R = pattern_size  # cols, rows
    if det.shape[0] != ref.shape[0] or det.shape[0] != C*R:
        return det  # fallback
    det_grid = det.reshape(R, C, 2)
    candidates = [
        det_grid,
        det_grid[:, ::-1, :],
        det_grid[::-1, :, :],
        det_grid[::-1, ::-1, :],
        np.transpose(det_grid, (1,0,2)),
        np.transpose(det_grid[:, ::-1, :], (1,0,2)),
        np.transpose(det_grid[::-1, :, :], (1,0,2)),
        np.transpose(det_grid[::-1, ::-1, :], (1,0,2)),
    ]
    ref_flat = ref.reshape(-1,2)
    best = det
    best_err = np.inf
    for cand in candidates:
        c_flat = cand.reshape(-1,2)
        err = float(np.mean(np.linalg.norm(c_flat - ref_flat, axis=1)))
        if err < best_err:
            best_err = err
            best = c_flat
    return best

def board_mask_outer_from_corners(image_shape: Tuple[int,int],
                                  inner_corners: np.ndarray,
                                  pattern_size: Tuple[int,int],
                                  expand: float = 0.5,
                                  feather_px: Optional[int] = None) -> np.ndarray:
    """
    Маска всей шахматной зоны, включая внешний ряд клеток.
    Строится из внутренних углов (R x C), экстраполируя сетку на expand клетки наружу.
    expand=0.5 -> классические пол-клетки до края доски.
    """
    H, W = image_shape[:2] if isinstance(image_shape, tuple) else image_shape
    C, R = pattern_size  # (cols, rows)
    pts = np.asarray(inner_corners, np.float32).reshape(-1,2)
    if pts.size != 2 * R * C:
        return np.zeros((H, W), np.uint8)

    G = pts.reshape(R, C, 2)

    hx_tl = G[0, 1] - G[0, 0]         ;  vy_tl = G[1, 0] - G[0, 0]
    hx_tr = G[0, -1] - G[0, -2]       ;  vy_tr = G[1, -1] - G[0, -1]
    hx_br = G[-1, -1] - G[-1, -2]     ;  vy_br = G[-1, -1] - G[-2, -1]
    hx_bl = G[-1, 1] - G[-1, 0]       ;  vy_bl = G[-1, 0] - G[-2, 0]

    TL = G[0, 0]     - expand*hx_tl - expand*vy_tl
    TR = G[0, -1]    + expand*hx_tr - expand*vy_tr
    BR = G[-1, -1]   + expand*hx_br + expand*vy_br
    BL = G[-1, 0]    - expand*hx_bl + expand*vy_bl

    poly = np.array([TL, TR, BR, BL], dtype=np.float32).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)

    if feather_px is None:
        D = np.sqrt(((pts[None,:,:]-pts[:,None,:])**2).sum(-1))
        np.fill_diagonal(D, np.inf)
        step = float(np.median(D.min(axis=1))) if np.isfinite(D).any() else 6.0
        feather_px = max(1, int(0.15 * step))
    if feather_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*feather_px+1, 2*feather_px+1))
        mask = cv2.erode(mask, k, iterations=1)   # слегка внутрь, чтобы не захватить рамку
    return mask


def outer_corners_from_inner(inner_corners: np.ndarray,
                             pattern_size: Tuple[int, int],
                             expand: float = 0.5) -> np.ndarray:
    """
    Оценивает 4 внешних угла всей доски (TL, TR, BR, BL) из внутренних R×C углов.
    expand=0.5 соответствует «пол-клетки до края».
    Возвращает массив shape (4,2) в порядке [TL, TR, BR, BL].
    """
    C, R = pattern_size  # cols, rows
    pts = np.asarray(inner_corners, np.float32).reshape(-1, 2)
    if pts.size != 2 * R * C:
        raise ValueError("inner_corners shape mismatch with pattern_size")

    G = pts.reshape(R, C, 2)

    hx_tl = G[0, 1]   - G[0, 0]     ; vy_tl = G[1, 0]   - G[0, 0]
    hx_tr = G[0, -1]  - G[0, -2]    ; vy_tr = G[1, -1]  - G[0, -1]
    hx_br = G[-1, -1] - G[-1, -2]   ; vy_br = G[-1, -1] - G[-2, -1]
    hx_bl = G[-1, 1]  - G[-1, 0]    ; vy_bl = G[-1, 0]  - G[-2, 0]

    TL = G[0, 0]     - expand*hx_tl - expand*vy_tl
    TR = G[0, -1]    + expand*hx_tr - expand*vy_tr
    BR = G[-1, -1]   + expand*hx_br + expand*vy_br
    BL = G[-1, 0]    - expand*hx_bl + expand*vy_bl

    return np.vstack([TL, TR, BR, BL]).astype(np.float32)
