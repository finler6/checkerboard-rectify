"calibrator/visual.py"
import os
from typing import Optional, Dict, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt

def _rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    return np.clip(g, 0, 255).astype(np.uint8)

def _scatter(ax, pts: Optional[np.ndarray], color="r", marker="o", size=20):
    if pts is None:
        return
    p = np.asarray(pts).reshape(-1, 2)
    ax.scatter(p[:, 0], p[:, 1], c=color, marker=marker, s=size)

def visualize_pipeline(
    original: np.ndarray,           # reference (cropped_ref)
    distorted: np.ndarray,          # distorted (cropped input)
    corrected: np.ndarray,          # corrected (cropped_corr)
    true_pts: Optional[np.ndarray],         # true corners in cropped coords
    detected_pts: Optional[np.ndarray],     # detected corners in cropped coords
    metrics: Optional[Dict[str, Any]] = None,
    out_path: Optional[str] = None,
    mask: Optional[np.ndarray] = None       # board mask in cropped coords (HxW, 0/255)
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(_rgb(distorted))
    _scatter(axs[0, 0], detected_pts, color="red", marker="o", size=18)
    axs[0, 0].set_title("Distorted (detected)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(_rgb(original))
    _scatter(axs[0, 1], true_pts, color="lime", marker="x", size=22)
    axs[0, 1].set_title("Reference (true)")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(_rgb(corrected))
    axs[1, 0].set_title("Corrected")
    axs[1, 0].axis("off")

    ref_g = _gray_u8(original)
    cor_g = _gray_u8(corrected)
    diff = cv2.absdiff(ref_g, cor_g)

    if mask is not None:
        m = (mask > 0)
        diff_vis = np.zeros_like(diff)
        diff_vis[m] = diff[m]
    else:
        diff_vis = diff

    axs[1, 1].imshow(diff_vis, cmap="gray")
    axs[1, 1].set_title("Absolute diff (gray)")
    axs[1, 1].axis("off")

    if metrics:
        mse = float(metrics.get("mse", 0.0))
        psnr = float(metrics.get("psnr", 0.0))
        ssim = float(metrics.get("ssim", 0.0))
        txt = f"mse: {mse:.4f}\npsnr: {psnr:.4f}\nssim: {ssim:.4f}"
        axs[1, 1].text(
            0.03, 0.97, txt, transform=axs[1, 1].transAxes,
            va="top", ha="left", fontsize=11,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=4)
        )

    plt.tight_layout(pad=1.2)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.show()
    plt.close(fig)
