"calibrator/prep.py"
import numpy as np
import cv2
from typing import Tuple

def generate_checkerboard(pattern_size: Tuple[int,int]=(7,7), square_size: int=40, noise: float=0.0):
    """
    Возвращает (img, true_corners)
    - img: BGR uint8 изображение
    - true_corners: numpy array shape (N,2) float32 с координатами внутренних углов (row-major)
    pattern_size = (cols, rows) of inner corners (OpenCV convention)
    """
    cols, rows = pattern_size
    width = cols * square_size + square_size
    height = rows * square_size + square_size

    img = 255 * np.ones((height, width), dtype=np.uint8)

    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0 = r * square_size
                x0 = c * square_size
                img[y0:y0 + square_size, x0:x0 + square_size] = 0

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if noise > 0:
        noise_m = (np.random.randn(*img.shape) * 255 * noise).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise_m, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)

    true_corners = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 1) * square_size
            y = (r + 1) * square_size
            true_corners.append([x, y])
    true_corners = np.array(true_corners, dtype=np.float32).reshape(-1, 2)

    return img_bgr, true_corners
