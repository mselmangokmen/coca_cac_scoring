"""PyTorch dataset over COCA volumes: one item is one patient's full series.

    ds = CocaVolumes("/data/coca", "gated", "splits/coca_gated_split.csv", sides=("train",))
    for item in volume_loader(ds, num_workers=4):
        item["hu"]          # (Z, 512, 512) int16, z-ascending
        item["spacing"], item["thickness"], item["agatston"], item["pid"], item["split"]

A volume is read whole because every downstream step (feature extraction, the scoring head, the
localization pool) works per volume; the DataLoader only parallelises the DICOM reading. Patients come
in the order of the split file (sorted by patient id), and that order is part of the protocol: the
feature caches, and hence the head's random draws, follow it.
"""
import os

import torch

from .cohorts import cohort_root, read_split, series_dirs
from .dicom_io import load_volume, parse_osirix_rois


class CocaVolumes(torch.utils.data.Dataset):
    """`sides`: which split sides to include ("train", "test"). `with_rois=True` (gated only) adds the
    parsed OsiriX contours as `rois` ({ImageIndex: [ROI]}). A series that cannot be read comes back
    with `error` set instead of `hu`, and the consumer skips it."""

    def __init__(self, data_root, cohort, split_csv, sides=("train", "test"), with_rois=False):
        if with_rois and cohort != "gated":
            raise ValueError("contours exist for the gated arm only")
        self.cohort, self.with_rois = cohort, with_rois
        dirs = series_dirs(data_root, cohort)
        rows = [r for r in read_split(split_csv) if r[1] in sides]
        missing = [pid for pid, _, _ in rows if pid not in dirs]
        if missing:
            raise FileNotFoundError(f"{len(missing)} patients of {split_csv} have no series under "
                                    f"{data_root} (first: {missing[:5]})")
        self.rows = sorted(rows)
        self.dirs = dirs
        self.xml_dir = os.path.join(cohort_root(data_root, cohort), "calcium_xml")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pid, split, score = self.rows[i]
        item = dict(pid=pid, split=split, agatston=score)
        try:
            hu, spacing, thickness = load_volume(self.dirs[pid])
            item.update(hu=hu, spacing=spacing, thickness=thickness)
            if self.with_rois:
                item["rois"] = parse_osirix_rois(os.path.join(self.xml_dir, f"{pid}.xml"))
        except Exception as e:                      # unreadable series: reported, not fatal
            item["error"] = f"{type(e).__name__}: {e}"
        return item


def _identity(x):
    return x


def volume_loader(ds, num_workers=4):
    """Items in dataset order, read by `num_workers` processes (no batching, no tensor conversion)."""
    return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers,
                                       collate_fn=_identity)
