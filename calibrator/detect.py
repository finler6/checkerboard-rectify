"calibrator/detect.py"
import cv2
import numpy as np
from typing import Optional, Tuple

def _preprocess_for_detection(gray: np.ndarray) -> np.ndarray:
    """CLAHE + optional blur to improve contrast / reduce small noise."""
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    return gray

def find_chessboard_corners(img: np.ndarray, pattern_size: Tuple[int, int]=(7,7)) -> Optional[np.ndarray]:
    """
    Попытка найти внутренние углы шахматной доски.
    Возвращает numpy array shape (N,2) dtype=float32 или None, если не найдено.
    pattern_size = (cols, rows)
    """
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy().astype(np.uint8)
    gray_proc = _preprocess_for_detection(gray)

    corners = None
    found = False
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE

    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            found_sb, corners_sb = cv2.findChessboardCornersSB(gray_proc, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE)
            if found_sb:
                corners = corners_sb.reshape(-1, 2)
                found = True
        except Exception:
            found = False

    if not found:
        found, corners = cv2.findChessboardCorners(gray_proc, pattern_size, flags)
        if found:
            corners = corners.reshape(-1, 2)

    if not found:
        h, w = gray_proc.shape[:2]
        scale = 0.5
        small = cv2.resize(gray_proc, (int(w*scale), int(h*scale)))
        found_small, corners_small = cv2.findChessboardCorners(small, pattern_size, flags)
        if found_small:
            corners = corners_small.reshape(-1, 2).astype(np.float32) / scale
            found = True

    if not found:
        return None

    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 1e-6)
    cv2.cornerSubPix(gray, corners, (7,7), (-1,-1), term)

    return corners.reshape(-1, 2).astype(np.float32)

def find_circles_grid_corners(img, pattern_size=(7,7), asymmetric=False):
    """
    Поиск узлов по круговой решётке (симметричной или асимметричной).
    Возвращает (N,2) или None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy().astype(np.uint8)
    flags = cv2.CALIB_CB_SYMMETRIC_GRID
    if asymmetric:
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID
    found, centers = cv2.findCirclesGrid(gray, pattern_size, flags)
    if not found:
        return None
    return centers.reshape(-1, 2).astype(np.float32)

