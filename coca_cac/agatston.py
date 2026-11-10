"""Agatston scoring, calcium masks from the OsiriX contours, and the CAC-DRS risk bands.

The gated COCA arm ships calcium contours but no per-patient score, so its reference score is
recomputed here from the contours (`agatston_from_masks`). The contours are HOLLOW outlines; each one
is closed, filled and dilated by 2 px (`fill_and_dilate`), and only pixels above 130 HU inside the
result count as calcium. Dilating before the HU cut recovers calcified pixels that a hand-drawn
outline clips, while the HU cut keeps the dilation from adding non-calcified tissue.

A LESION IS A CONNECTED COMPONENT (8-connectivity) of `mask & HU > 130` on one slice; lesions of
1 mm^2 or less are discarded; each lesion contributes area (mm^2) x density weight (1-4 by peak HU).

`agatston_from_rois` is an independent cross-check: per ROI, no dilation, filled contour & HU >= 130.
It reproduces the annotator's own OsiriX `Area` / `Max` values, which validates the contour filling
and the ImageIndex mapping. Over the cohort the two definitions agree closely (median ratio ~1.0).
"""
import numpy as np
from matplotlib.path import Path as _MplPath

HU_THRESHOLD = 130.0      # the Agatston calcium threshold
MIN_AREA_MM2 = 1.0        # lesions of this area or less are discarded
DILATION_PX = 2

CAC_DRS = ("0-10", "11-100", "101-400", "400+")


def cac_drs(score):
    """Risk band of an Agatston score: <=10, 11-100, 101-400, >400."""
    if score <= 10:
        return "0-10"
    if score <= 100:
        return "11-100"
    if score <= 400:
        return "101-400"
    return "400+"


def agatston_weight(peak_hu):
    """Density weight of a lesion from its peak HU (0 below the calcium threshold)."""
    if peak_hu >= 400.0:
        return 4
    if peak_hu >= 300.0:
        return 3
    if peak_hu >= 200.0:
        return 2
    if peak_hu >= HU_THRESHOLD:
        return 1
    return 0


def fill_and_dilate(pixels, shape, dilation_px=DILATION_PX):
    """Hollow contour points -> filled, dilated boolean mask: morphological close (9x9 ellipse, 5
    iterations), hole filling, polygon filling of the outer contours, grey dilation by `dilation_px`.
    The HU cut is applied by the caller, after the dilation."""
    import cv2
    import scipy.ndimage
    m = np.zeros(shape, np.uint8)
    q = np.asarray(pixels, int)
    m[np.clip(q[:, 1], 0, shape[0] - 1), np.clip(q[:, 0], 0, shape[1] - 1)] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=5)
    m = scipy.ndimage.binary_fill_holes(m).astype(np.uint8)
    for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        cv2.fillPoly(m, [c], 1)
    if dilation_px > 0:
        r = dilation_px
        m = scipy.ndimage.grey_dilation(m, size=(2 * r + 1, 2 * r + 1))
    return m.astype(bool)


def annotation_mask(rois, shape, dilation_px=DILATION_PX):
    """Union of the filled, dilated contours of one slice (all arteries together)."""
    m = np.zeros(shape, bool)
    for roi in rois:
        m |= fill_and_dilate(roi["pixels"], shape, dilation_px)
    return m


def calcium_mask(hu_slice, rois, dilation_px=DILATION_PX):
    """Pixel-level calcium mask of one slice: annotated (filled, dilated) AND HU > 130. This is the
    mask the reference score is computed from and the mask the localization metrics use."""
    return annotation_mask(rois, hu_slice.shape, dilation_px) & (hu_slice > HU_THRESHOLD)


def agatston_score_single_channel(image, mask, spacing, threshold=HU_THRESHOLD):
    """Agatston contribution of one slice: connected components (8-connectivity) of
    `mask & image > threshold`, each with area (mm^2, from the isotropic pixel `spacing`) above 1 mm^2,
    times its density weight."""
    from skimage import measure
    calcified = np.logical_and(image > threshold, mask > 0)
    if not calcified.any():
        return 0.0
    labels = measure.label(calcified, connectivity=2)
    total = 0.0
    for prop in measure.regionprops(labels, intensity_image=image):
        area = prop.area * (spacing * spacing)
        if area <= MIN_AREA_MM2:
            continue
        total += area * agatston_weight(prop.intensity_max)
    return float(total)


def agatston_from_masks(hu, rois_by_slice, spacing, dilation_px=DILATION_PX, return_slices=False):
    """Reference Agatston score of one gated volume from its contours.

    `hu` is the z-ascending HU stack and `rois_by_slice` the parsed OsiriX file ({ImageIndex: [ROI]}).
    Returns (total, out_of_range) where out_of_range counts ROIs whose ImageIndex falls outside the
    stack; a non-zero count means the score of that patient cannot be trusted and it is excluded."""
    total, per_slice, out_of_range = 0.0, {}, 0
    for iidx, rois in rois_by_slice.items():
        if not (0 <= iidx < len(hu)):
            out_of_range += len(rois)
            continue
        sl = hu[iidx]
        m = annotation_mask(rois, sl.shape, dilation_px)
        s = agatston_score_single_channel(sl.astype(np.float32), m, float(spacing))
        total += s
        if return_slices and s:
            per_slice[int(iidx)] = s
    return (total, per_slice, out_of_range) if return_slices else (total, out_of_range)


def fill_polygon(pixels, shape):
    """Contour points -> filled mask with a polygon fill (points re-ordered by angle about their
    centroid, the boundary itself included). Used only by the cross-check."""
    p = np.asarray(pixels, dtype=float)
    if p.ndim != 2 or p.shape[0] < 3:
        m = np.zeros(shape, bool)
        if p.size:
            q = p.astype(int)
            m[np.clip(q[:, 1], 0, shape[0] - 1), np.clip(q[:, 0], 0, shape[1] - 1)] = True
        return m
    c = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    m = _MplPath(p).contains_points(np.c_[xx.ravel(), yy.ravel()], radius=0.5).reshape(shape)
    q = np.asarray(pixels, dtype=int)
    m[np.clip(q[:, 1], 0, shape[0] - 1), np.clip(q[:, 0], 0, shape[1] - 1)] = True
    return m


def agatston_from_rois(hu, rois_by_slice, spacing):
    """Cross-check score: per ROI, no dilation, filled contour & HU >= 130, one lesion per ROI.
    Returns (total, out_of_range)."""
    px_area = float(spacing) ** 2
    total, out_of_range = 0.0, 0
    for iidx, rois in rois_by_slice.items():
        if not (0 <= iidx < len(hu)):
            out_of_range += len(rois)
            continue
        sl = hu[iidx]
        for roi in rois:
            m = fill_polygon(roi["pixels"], sl.shape) & (sl >= HU_THRESHOLD)
            area = int(m.sum()) * px_area
            if area < MIN_AREA_MM2:
                continue
            total += area * agatston_weight(float(sl[m].max()))
    return total, out_of_range
