import numpy as np
from scipy.ndimage import center_of_mass, generic_filter, label
from skimage.filters import threshold_otsu


def otsu_with_gate(
    image: np.ndarray, fallback_threshold: float, is_sar: bool = False
) -> tuple[str, float]:
    """
    Computes Otsu threshold with a bimodality gate.
    If the between-class variance ratio >= 0.5 and both classes >= 5% of valid pixels,
    returns ("otsu", threshold). Otherwise returns ("fixed_fallback", fallback_threshold).
    """
    valid = image[np.isfinite(image)]
    if valid.size == 0:
        return "fixed_fallback", fallback_threshold

    # Calculate Otsu
    try:
        thresh = float(threshold_otsu(valid))
    except ValueError:
        # e.g., all values are the same
        return "fixed_fallback", fallback_threshold

    # Bimodality gate checks
    # 1. Variance ratio = between-class variance / total variance
    mask = valid >= thresh
    class1 = valid[~mask]
    class2 = valid[mask]

    n_total = valid.size
    w1 = class1.size / n_total
    w2 = class2.size / n_total

    if w1 < 0.05 or w2 < 0.05:
        # Fails class size gate
        return "fixed_fallback", fallback_threshold

    var_total = np.var(valid)
    if var_total == 0:
        return "fixed_fallback", fallback_threshold

    mean1 = np.mean(class1)
    mean2 = np.mean(class2)
    mean_total = np.mean(valid)

    var_between = w1 * (mean1 - mean_total) ** 2 + w2 * (mean2 - mean_total) ** 2
    ratio = var_between / var_total

    if ratio >= 0.5:
        return "otsu", thresh
    return "fixed_fallback", fallback_threshold


def get_largest_component_centroid(binary_mask: np.ndarray) -> tuple[float, float] | None:
    """
    Finds the largest connected component in a binary mask and returns its
    normalised (x, y) centroid. Returns None if no pixels are True.
    """
    if not np.any(binary_mask):
        return None

    labeled, num_features = label(binary_mask)
    if num_features == 0:
        return None

    # Find largest component (ignoring background 0)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()

    # Calculate centroid of the largest component
    # center_of_mass returns (y, x) in array indices
    y, x = center_of_mass(binary_mask, labeled, largest_label)

    # Normalise to [0, 1] relative to image dimensions
    h, w = binary_mask.shape
    return (float(x / w), float(y / h))


def compute_texture(image: np.ndarray) -> np.ndarray:
    """
    Calculates a local texture map (local standard deviation).
    This serves as a fast approximation for texture/morphology branches.
    """
    # A simple 3x3 local std filter
    return generic_filter(image.astype(float), np.std, size=3)
