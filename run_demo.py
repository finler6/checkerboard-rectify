"run_demo.py"

from calibrator.prep import generate_checkerboard
from calibrator.detect import find_chessboard_corners
from calibrator.calc import (
    apply_transform, compute_metrics, compute_metrics_masked,
    match_points_hungarian, match_points_greedy,
    refine_homography_leastsq, choose_best_transform,
    reorder_chessboard_to_reference,
    board_mask_from_corners
)
import numpy as np
import cv2
import argparse
import yaml


def make_affine_transform(center, angle_deg=12.3, scale=0.92, tx=18, ty=-12):
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def reprojection_error(M, src_pts, dst_pts):
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if M.shape == (2, 3):
        src_h = np.hstack([src, np.ones((len(src), 1), dtype=np.float64)])
        proj = (M @ src_h.T).T
    else:
        src_h = np.hstack([src, np.ones((len(src), 1), dtype=np.float64)])
        p = (M @ src_h.T).T
        proj = p[:, :2] / p[:, 2:3]
    d = np.linalg.norm(proj - dst, axis=1)
    return d, float(d.mean()), float(d.std())


def _parse_pattern(s: str):
    a, b = s.lower().split("x")
    return (int(a), int(b))


def _estimate_nn_step(pts: np.ndarray) -> float:
    P = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(P) < 2:
        return 40.0
    D2 = (P[:, None, :] - P[None, :, :]) ** 2
    D = np.sqrt(D2.sum(axis=2))
    np.fill_diagonal(D, np.inf)
    nn = np.min(D, axis=1)
    return float(np.median(nn))


def _estimate_checker_parity(gray_img: np.ndarray, ref_corners: np.ndarray, pattern):
    cols, rows = pattern  # (cols, rows)
    corners = ref_corners.reshape(rows, cols, 2)
    centers, labels = [], []
    for r in range(rows - 1):
        for c in range(cols - 1):
            p = corners[r:r+2, c:c+2, :].mean(axis=(0, 1))
            centers.append(p)
            labels.append((r + c) % 2)
    centers = np.asarray(centers, np.float32)
    labels = np.asarray(labels, np.int32)
    H, W = gray_img.shape[:2]
    xs = np.clip(np.round(centers[:, 0]).astype(int), 0, W-1)
    ys = np.clip(np.round(centers[:, 1]).astype(int), 0, H-1)
    vals = gray_img[ys, xs].astype(np.float32)
    black_mean = vals[labels == 0].mean()
    white_mean = vals[labels == 1].mean()
    return +1 if (white_mean - black_mean) >= 0 else -1


def _inner_board_mask_from_corners(image_shape, inner_corners: np.ndarray, shrink_ratio: float = 0.0):
    H, W = image_shape[:2]
    pts = np.asarray(inner_corners, np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return np.zeros((H, W), np.uint8)
    hull = cv2.convexHull(pts).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if shrink_ratio > 0:
        D = np.sqrt(((pts[None, :, :] - pts[:, None, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        nn = np.median(D.min(axis=1)) if np.isfinite(D).any() else 8.0
        rad = max(2, int(shrink_ratio * nn))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rad+1, 2*rad+1))
        mask = cv2.erode(mask, k, iterations=1)
    return mask

def _cell_centers_mask_from_corners(image_shape, inner_corners: np.ndarray, pattern, frac: float = 0.36):
    H, W = image_shape[:2]
    pts = np.asarray(inner_corners, np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return np.zeros((H, W), np.uint8)
    cols, rows = pattern
    G = pts.reshape(rows, cols, 2)
    mask = np.zeros((H, W), np.uint8)
    for r in range(rows - 1):
        for c in range(cols - 1):
            p00 = G[r, c]
            p01 = G[r, c+1]
            p10 = G[r+1, c]
            p11 = G[r+1, c+1]
            center = (p00 + p01 + p10 + p11) * 0.25
            vx = p01 - p00
            vy = p10 - p00
            u = vx * frac
            v = vy * frac
            poly = np.array([
                center - u - v,
                center + u - v,
                center + u + v,
                center - u + v
            ], dtype=np.float32).astype(np.int32)
            cv2.fillConvexPoly(mask, poly, 255)
    return mask

def _idealize_checker(gray_img: np.ndarray, ref_corners: np.ndarray, pattern, frac: float = 0.36) -> np.ndarray:
    H, W = gray_img.shape[:2]
    cols, rows = pattern
    pts = np.asarray(ref_corners, np.float32).reshape(-1, 2)
    G = pts.reshape(rows, cols, 2)
    cell_stats = []
    cell_polys = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            p00 = G[r, c]
            p01 = G[r, c+1]
            p10 = G[r+1, c]
            p11 = G[r+1, c+1]
            center = (p00 + p01 + p10 + p11) * 0.25
            vx = p01 - p00
            vy = p10 - p00
            u = vx * frac
            v = vy * frac
            poly = np.array([
                center - u - v,
                center + u - v,
                center + u + v,
                center - u + v
            ], dtype=np.float32).astype(np.int32)
            cell_polys.append(poly)
            mask = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(mask, poly, 255)
            vals = gray_img[mask > 0]
            if vals.size == 0:
                m = 127.0
            else:
                m = float(np.median(vals))
            cell_stats.append(m)
    arr = np.asarray(cell_stats, np.float32)
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    _, thr = cv2.threshold(arr_u8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(_)
    bright = arr > t
    ref_labels = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            ref_labels.append(((r + c) % 2) == 1)
    ref_labels = np.array(ref_labels, dtype=bool)
    agree = int(np.sum(bright == ref_labels))
    disagree = bright.size - agree
    if disagree > agree:
        bright = ~bright
    out = np.zeros((H, W), np.uint8)
    idx = 0
    for poly in cell_polys:
        col = 255 if bright[idx] else 0
        cv2.fillConvexPoly(out, poly, int(col))
        idx += 1
    return out


def main(use_homography=True, use_hungarian=True, crop_ratio=0.1, out_path="demo_result.png", cfg_path="config.yaml"):
    global args

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f) or {}
    except Exception:
        CFG = {}

    pattern = _parse_pattern(CFG.get("pattern", {}).get("size", getattr(args, "pattern", "7x7")))

    input_path = getattr(args, "input", None)
    pad = int(CFG.get("visual", {}).get("pad", 80))

    if input_path:
        distorted_raw = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if distorted_raw is None:
            print(f"Не удалось открыть файл: {input_path}")
            return
        h, w = distorted_raw.shape[:2]
        distorted_img = cv2.copyMakeBorder(
            distorted_raw, pad, pad, pad, pad,
            borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        detected = find_chessboard_corners(distorted_img, pattern_size=pattern)
        if detected is None:
            print("Не удалось найти углы на искаженном изображении.")
            return

        est_step = int(round(_estimate_nn_step(detected)))
        est_step = max(8, min(est_step, 200))
        ref_board, ref_pts = generate_checkerboard(pattern_size=pattern, square_size=est_step, noise=0.0)

        H_ref, W_ref = ref_board.shape[:2]
        ref_canvas = 255 * np.ones((h, w, 3), dtype=np.uint8)
        y0 = (h - H_ref) // 2
        x0 = (w - W_ref) // 2
        ref_canvas[y0:y0+H_ref, x0:x0+W_ref] = ref_board
        true_corners = (ref_pts + np.array([x0, y0], np.float32)).astype(np.float32)

        true_corners_shifted = true_corners + np.array([pad, pad], np.float32)
        canvas_ref = cv2.copyMakeBorder(ref_canvas, pad, pad, pad, pad,
                                        borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
        Hc, Wc = canvas_ref.shape[:2]
    else:
        square_size = int(getattr(args, "square", 40))
        ref_board, ref_pts = generate_checkerboard(pattern_size=pattern, square_size=square_size, noise=0.0)
        H_ref, W_ref = ref_board.shape[:2]
        canvas = 255 * np.ones((H_ref + 2*pad, W_ref + 2*pad, 3), dtype=np.uint8)
        canvas[pad:pad+H_ref, pad:pad+W_ref] = ref_board
        true_corners = ref_pts
        true_corners_shifted = true_corners + np.array([pad, pad], np.float32)
        M_true = make_affine_transform((canvas.shape[1] / 2, canvas.shape[0] / 2))
        distorted_img = cv2.warpAffine(canvas, M_true, (canvas.shape[1], canvas.shape[0]),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        canvas_ref = canvas.copy()
        Hc, Wc = canvas_ref.shape[:2]
        detected = find_chessboard_corners(distorted_img, pattern_size=pattern)
        if detected is None:
            print("Не удалось найти углы на синтетическом изображении.")
            return

    try:
        detected_ordered = reorder_chessboard_to_reference(detected, pattern, true_corners_shifted)
        detected_matched = detected_ordered if detected_ordered.shape == true_corners_shifted.shape else None
    except Exception:
        detected_matched = None

    if detected_matched is None:
        if use_hungarian:
            try:
                detected_matched = match_points_hungarian(detected, true_corners_shifted)
            except Exception as e:
                print("Hungarian failed, fallback to greedy:", e)
                detected_matched = match_points_greedy(detected, true_corners_shifted)
        else:
            detected_matched = match_points_greedy(detected, true_corners_shifted)

    print("First 5 pairs (detected -> true):")
    for i in range(min(5, len(detected_matched))):
        print(f"detected: {detected_matched[i]}  ->  true: {true_corners_shifted[i]}")

    M0, method, used_idx = choose_best_transform(detected_matched, true_corners_shifted, ransac_thresh=3.0)
    print(f"Selected model: {method}  |  inliers: {len(used_idx)}/{len(detected_matched)}")
    if method == "homography" and len(used_idx) >= 4:
        M_ref, _ = refine_homography_leastsq(M0, detected_matched[used_idx], true_corners_shifted[used_idx], verbose=True)
        _, mean0, _ = reprojection_error(M0, detected_matched, true_corners_shifted)
        _, mean1, _ = reprojection_error(M_ref, detected_matched, true_corners_shifted)
        M_est = M_ref if mean1 <= mean0 * 1.05 else M0
    else:
        M_est = M0

    corrected_full = apply_transform(distorted_img, M_est, (Wc, Hc), borderMode=cv2.BORDER_CONSTANT)

    if input_path:
        cropped_ref = canvas_ref[pad:-pad, pad:-pad]
        cropped_corr = corrected_full[pad:-pad, pad:-pad]
        true_in_cropped = (true_corners_shifted - np.array([pad, pad], np.float32))
        distorted_crop = distorted_img[pad:-pad, pad:-pad]
        detected_for_vis = detected - np.array([pad, pad], dtype=np.float32)
    else:
        cropped_ref = canvas_ref[pad:pad+H_ref, pad:pad+W_ref]
        cropped_corr = corrected_full[pad:pad+H_ref, pad:pad+W_ref]
        true_in_cropped = ref_pts
        distorted_crop = distorted_img[pad:pad+H_ref, pad:pad+W_ref]
        detected_for_vis = detected - np.array([pad, pad], dtype=np.float32)

    metrics_full = compute_metrics(cropped_ref, cropped_corr)

    board_mask = board_mask_from_corners(cropped_ref.shape, true_in_cropped)
    frac = float(CFG.get("ssim", {}).get("cell_frac", 0.32))
    cell_mask = _cell_centers_mask_from_corners(cropped_ref.shape, true_in_cropped, pattern, frac=frac)
    parity = _estimate_checker_parity(cv2.cvtColor(cropped_corr, cv2.COLOR_BGR2GRAY), true_in_cropped, pattern)
    corr_for_metrics = (255 - cropped_corr) if parity < 0 else cropped_corr
    ssim_cfg = CFG.get("ssim", {})
    metrics_mask = compute_metrics_masked(
        cropped_ref, corr_for_metrics, cell_mask,
        photometric=bool(ssim_cfg.get("photometric", True)),
        blur_sigma=float(ssim_cfg.get("blur_sigma", 2.0)),
        downscale_factor=float(ssim_cfg.get("use_downscale", True) and ssim_cfg.get("downscale_factor", 0.75) or 1.0),
        photometric_method=str(ssim_cfg.get("photometric_method", "linear"))
    )

    ideal_bin = _idealize_checker(cv2.cvtColor(cropped_corr, cv2.COLOR_BGR2GRAY), true_in_cropped, pattern, frac=0.36)
    ref_gray = cv2.cvtColor(cropped_ref, cv2.COLOR_BGR2GRAY)
    _, ref_bin = cv2.threshold(ref_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    from calibrator.calc import compute_metrics_binary_masked
    metrics_bin = compute_metrics_binary_masked(ref_bin, ideal_bin, board_mask)

    d_all, mean_all, std_all = reprojection_error(M_est, detected_matched, true_corners_shifted)
    print(f"Reprojection (ALL) — mean: {mean_all:.3f}, std: {std_all:.3f}, max: {d_all.max():.3f}")
    print("Metrics full:", metrics_full)
    print(f"Checker parity: {'OK' if parity>0 else 'INVERTED'}")
    print("Metrics masked (board only):", metrics_mask)
    print("Binary pattern metrics:", metrics_bin)

    try:
        from calibrator.visual import visualize_pipeline
        viz_metrics = {"mse": metrics_mask["mse"], "psnr": metrics_mask["psnr"], "ssim": metrics_mask["ssim"]}
        visualize_pipeline(
            cropped_ref, distorted_crop, cropped_corr,
            true_in_cropped, detected_for_vis,
            metrics=viz_metrics, out_path=out_path,
            mask=board_mask
        )
    except Exception as e:
        print("Visualization failed:", e)
        cv2.imwrite("distorted.png", distorted_crop)
        cv2.imwrite("corrected.png", cropped_corr)
        print("Saved distorted.png and corrected.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-homography", action="store_true")
    parser.add_argument("--no-hungarian", action="store_true")
    parser.add_argument("--crop", type=float, default=0.1)
    parser.add_argument("--input", type=str, default=None, help="Путь к входному изображению для проверки")
    parser.add_argument("--mode", type=str, default="chess", choices=["chess", "circles"], help="Тип шаблона")
    parser.add_argument("--asymmetric", action="store_true", help="Асимметричная круговая решетка для --mode=circles")
    parser.add_argument("--pattern", type=str, default="7x7", help="Размер внутренней сетки, напр. 7x7")
    parser.add_argument("--square", type=int, default=40, help="(используется только в синтетике)")
    args = parser.parse_args()
    main(use_homography=not args.no_homography, use_hungarian=not args.no_hungarian, crop_ratio=args.crop)
