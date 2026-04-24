import numpy as np
RNG = np.random.default_rng(42)
IMG_SHAPE = (128, 128)

def flat_image(value: float, noise: float=0.0, shape=IMG_SHAPE) -> np.ndarray:
    img = np.full(shape, value, dtype=np.float32)
    if noise > 0:
        img += RNG.normal(0, noise, shape).astype(np.float32)
    return img

def gaussian_star(cx: float, cy: float, peak: float, sigma: float=3.0, shape=IMG_SHAPE) -> np.ndarray:
    y, x = np.mgrid[:shape[0], :shape[1]]
    return (peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)

def stack(n: int, value: float, noise: float=5.0) -> np.ndarray:
    return np.stack([flat_image(value, noise) for _ in range(n)], axis=0)
