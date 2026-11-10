"""DICOM series and OsiriX ROI reading for the Stanford COCA release.

Both COCA arms (gated and non-gated) ship one axial series per patient as a directory of DICOM files.
The gated arm also ships the annotators' calcium contours as OsiriX plist files (`calcium_xml/<pid>.xml`).

Conventions that the rest of the package relies on:
  * slices are sorted by ImagePositionPatient[2], ascending (z-ascending);
  * the OsiriX `ImageIndex` of an ROI indexes exactly that z-ascending order (verified on the release:
    under this mapping the ROIs land on tissue at or above 130 HU; under the reversed order they do not);
  * HU = stored value * RescaleSlope + RescaleIntercept, as int16.
"""
import ast
import os
import plistlib

import numpy as np
import pydicom

# OsiriX ROI names used in the release -> artery index
ARTERY_ID = {"Right Coronary Artery": 0, "Left Anterior Descending Artery": 1,
             "Left Coronary Artery": 2, "Left Circumflex Artery": 3}
DEFAULT_THICKNESS_MM = 3.0
DEFAULT_PIXEL_SPACING_MM = 0.625


def load_scan(series_dir):
    """Read every DICOM file of one series directory, sorted z-ascending.

    Returns (slices, pixel_spacing_mm, slice_thickness_mm). The pixel spacing is PixelSpacing[0]
    (isotropic in-plane in this release), falling back to ImagerPixelSpacing[0] and then to 0.625 mm.

    The slice thickness is the MEDIAN of the non-zero gaps between consecutive z positions. The gap
    between the first two slices alone is not robust: a duplicated position makes it 0.0 mm for some
    gated series, and the scoring head multiplies its prediction by thickness / 3. Fallbacks, used only
    when a series has no non-zero gap: the SliceThickness tag, then 3.0 mm."""
    slices = [pydicom.dcmread(os.path.join(series_dir, f)) for f in os.listdir(series_dir)]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    try:
        spacing = float(slices[0].PixelSpacing[0])
    except Exception:
        try:
            spacing = float(slices[0].ImagerPixelSpacing[0])
        except Exception:
            spacing = DEFAULT_PIXEL_SPACING_MM

    thickness = float("nan")
    z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    if len(z) > 1:
        gaps = np.abs(np.diff(z))
        gaps = gaps[gaps > 1e-3]
        if gaps.size:
            thickness = float(np.median(gaps))
    if not thickness > 0:
        try:
            thickness = float(slices[0].SliceThickness)
        except Exception:
            thickness = DEFAULT_THICKNESS_MM
    if not thickness > 0:
        thickness = DEFAULT_THICKNESS_MM
    return slices, spacing, thickness


def get_pixels_hu(slices):
    """(Z, H, W) int16 Hounsfield units from the stored pixels and the rescale tags. Stored -2000
    (outside the reconstruction circle) is set to 0 before rescaling."""
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    image[image == -2000] = 0
    for i, s in enumerate(slices):
        intercept, slope = s.RescaleIntercept, s.RescaleSlope
        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.int16)
        image[i] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def find_series_dir(path):
    """Descend through directories that hold exactly one entry, itself a directory (e.g. <pid>/<series>/)."""
    while True:
        inner = os.listdir(path)
        if len(inner) == 1 and os.path.isdir(os.path.join(path, inner[0])):
            path = os.path.join(path, inner[0])
        else:
            return path


def load_volume(series_dir):
    """-> (hu int16 (Z, H, W) z-ascending, pixel_spacing_mm, slice_thickness_mm)."""
    slices, spacing, thickness = load_scan(series_dir)
    return get_pixels_hu(slices), spacing, thickness


def _point(s):
    """OsiriX writes a pixel point as the string "(x, y)"; it is truncated to integer pixel indices."""
    x, y = ast.literal_eval(s)[:2]
    return int(x), int(y)


def parse_osirix_rois(xml_path):
    """OsiriX plist -> {ImageIndex: [ROI, ...]}.

    Each ROI is a dict with `cid` (artery index, see ARTERY_ID), `pixels` (the contour's distinct
    integer (x, y) points, unordered), and the annotator's own `Area` (cm^2) and `Max` (HU), which
    only serve as a cross-check. ROIs without points are skipped. An ROI name outside ARTERY_ID
    raises KeyError, so a malformed annotation fails loudly instead of being scored partially."""
    with open(xml_path, "rb") as f:
        pl = plistlib.load(f)
    rois = {}
    for image in pl["Images"]:
        iidx = int(image["ImageIndex"])
        for roi in image["ROIs"]:
            if not roi["Point_px"]:
                continue
            rois.setdefault(iidx, []).append(dict(
                cid=ARTERY_ID[roi["Name"]],
                pixels=list({_point(p) for p in roi["Point_px"]}),
                Area=roi.get("Area"), Max=roi.get("Max")))
    return rois
