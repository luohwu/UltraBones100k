import os

import numpy as np
from PIL import Image
from phasepack import phasesym

def phase_symmetry(img) -> np.ndarray:
    """
    Phase Symmetry (PS) as used in:
    Hacihaliloglu et al., 2009 (bone surface localization in ultrasound).

    Paper settings:
      - nscale (m) = 2
      - norient (Nr) = 6
      - minWaveLength (lmin) = 25 pixels
      - angular overlap s = 1.2  (paper states 25° bandwidth corresponds to s=1.2)
      - noise threshold multiplier k = 8

    Returns: float32 PS map.
    """
    # Convert to float grayscale in [0,1]
    if img.max()>1:
        I = np.asarray(img, dtype=np.float32) / 255.0
    else:
        I=img
    try:
        from phasepack import phasesym
    except ImportError as e:
        raise ImportError("Please install phasepack: pip install phasepack") from e

    # Key paper params
    nscale = 2
    norient = 6
    minWaveLength = 25
    k_noise = 0.1

    # phasesym returns (PS, orientation, ...) depending on version
    res = phasesym(
        I,
        nscale=nscale,
        norient=norient,
        minWaveLength=minWaveLength,
        mult=2.1,              # typical; paper doesn't specify explicit mult
        sigmaOnf=0.55*1,         # typical; paper specifies bandwidth via k/u0, not directly sigmaOnf
        k=k_noise,             # noise threshold multiplier
        polarity=0,            # detect both bright/dark symmetry (ridge-like bone response)
        noiseMethod=-1,        # Kovesi style noise estimation from smallest scale

    )

    # Extract PS map robustly
    if isinstance(res, (tuple, list)):
        ps = res[0]
    else:
        ps = res

    return ps.astype(np.float32)


def to_u8_for_cv(ps: np.ndarray) -> np.ndarray:
    """Normalize float PS -> uint8 [0,255] for display."""
    ps = np.nan_to_num(ps, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mn, mx = float(ps.min()), float(ps.max())
    print(f"PS min={mn:.6g}, max={mx:.6g}")
    if mx - mn < 1e-12:
        return np.zeros(ps.shape, dtype=np.uint8)
    ps01 = (ps - mn) / (mx - mn)
    return (ps01 * 255).astype(np.uint8)

def local_graylevel_scurve(
    img: np.ndarray,
    block_frac: float = 0.10,
    alpha: float = 0.9642,
    beta: float = 8.594e-4,
    gamma: float = 0.4962,
    delta: float = 0.07598,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Local gray-level S-curve transformation (Gandhamal et al., 2017).

    Input:
      - img: numpy array, grayscale, shape (H, W), float in [0, 1]
    Output:
      - numpy array, shape (H, W), float32 in [0, 1]
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array (H, W); got shape {img.shape}")

    # work in float for stability
    a = img.astype(np.float32, copy=False)

    # (Optional) clamp in case the caller slightly violates the contract
    a = np.clip(a, 0.0, 1.0)

    h, w = a.shape

    # blocks-per-dimension ~ block_frac of image size
    nbx = max(1, int(round(block_frac * w)))
    nby = max(1, int(round(block_frac * h)))
    bx = int(np.ceil(w / nbx))
    by = int(np.ceil(h / nby))

    out = np.empty_like(a, dtype=np.float32)

    for y0 in range(0, h, by):
        y1 = min(h, y0 + by)
        for x0 in range(0, w, bx):
            x1 = min(w, x0 + bx)

            block = a[y0:y1, x0:x1]
            Lmin = float(block.min())
            Lmax = float(block.max())
            rng = Lmax - Lmin

            if rng < eps:
                out[y0:y1, x0:x1] = block
                continue

            r = (block - Lmin) / (rng + eps)  # normalize to [0,1]

            # S-curve
            s = alpha + (beta - alpha) / (1.0 + np.exp((r - gamma) / (delta + eps)))

            out[y0:y1, x0:x1] = s * rng + Lmin  # back to [Lmin, Lmax]

    # keep output in [0, 1]
    return np.clip(out, 0.0, 1.0)

if __name__ == "__main__":
    import os
    import cv2
    from PIL import Image
    import numpy as np

    img_folder = "./../AI_ultrasound_segmentation/example_ultrasound_images"

    for file in os.listdir(img_folder):
        path = os.path.join(img_folder, file)

        img = Image.open(path).convert("L")
        img_enh = local_graylevel_scurve(np.asarray(img)/255.)

        ps = phase_symmetry(np.asarray(img))  # uses paper params by default
        ps_u8 = to_u8_for_cv(ps)

        img_np = np.array(img)
        enh_np = np.array(img_enh)

        # Stack: Original | Enhanced | PhaseSym
        side_by_side = np.hstack([img_np, to_u8_for_cv(enh_np), ps_u8])
        side_by_side = cv2.resize(
            side_by_side, None,
            fx=0.5, fy=0.5,
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("Original | Enhanced | Phase Symmetry", side_by_side)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 27:  # ESC to quit
            break