"""The two Stanford COCA arms: where the series are, where the reference scores come from, and the
patient-level split files.

Layout of the release (Stanford AIMI, "COCA - Coronary Calcium and Chest CTs"):

    <root>/Gated_release_final/
        patient/<pid>/<series>/*.dcm     ECG-gated non-contrast calcium-scoring CT
        calcium_xml/<pid>.xml            OsiriX calcium contours (no score is shipped)
    <root>/deidentified_nongated/
        <pid>/*.dcm  or  <pid>/<series>/*.dcm
        scores.xlsx                      per-patient total Agatston score (no contours are shipped)

The gated arm has contours but no scores, the non-gated arm scores but no contours. The gated reference
score is therefore recomputed from the contours (scripts/build_gated_reference.py), and only the gated
arm supports the localization (pointing) analysis.
"""
import csv
import os

GATED_DIR = "Gated_release_final"
NONGATED_DIR = "deidentified_nongated"
COHORTS = ("gated", "nongated")


def cohort_root(data_root, cohort):
    """`data_root` may be the release root or already the arm's own directory."""
    sub = {"gated": GATED_DIR, "nongated": NONGATED_DIR}[cohort]
    d = os.path.join(data_root, sub)
    return d if os.path.isdir(d) else data_root


def gated_series(gated_root):
    """[(pid, series_dir, xml_path)] for every gated patient with both a series and a contour file,
    sorted by patient id. A patient directory holding several series uses the first in sorted order
    (every annotated patient of the release holds one)."""
    pdir, xdir = os.path.join(gated_root, "patient"), os.path.join(gated_root, "calcium_xml")
    out = []
    for name in os.listdir(pdir):
        xml, d = os.path.join(xdir, f"{name}.xml"), os.path.join(pdir, name)
        if not (name.isdigit() and os.path.isfile(xml) and os.path.isdir(d)):
            continue
        series = sorted(s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s)))
        if series:
            out.append((int(name), os.path.join(d, series[0]), xml))
    return sorted(out)


def nongated_reference(nongated_root):
    """{pid: total Agatston} from the shipped scores.xlsx. Its id column is literally named
    "filename " (with a trailing space) and holds "A<pid>"."""
    import pandas as pd
    df = pd.read_excel(os.path.join(nongated_root, "scores.xlsx"))
    pid = df["filename "].astype(str).str.replace("A", "", regex=False).astype(int)
    return dict(zip(pid.tolist(), df["total"].astype(float).tolist()))


def nongated_series(nongated_root):
    """{pid: series_dir} for every scored non-gated patient whose directory exists. Some patients nest
    the series one level deeper (<pid>/<series>/*.dcm); that is detected by a single entry without an
    extension."""
    out = {}
    for pid in sorted(nongated_reference(nongated_root)):
        d = os.path.join(nongated_root, str(pid))
        if not os.path.isdir(d):
            continue
        inner = os.listdir(d)
        if len(inner) == 1 and "." not in inner[0]:
            d = os.path.join(d, inner[0])
        out[pid] = d
    return out


def series_dirs(data_root, cohort):
    """{pid: series_dir} for one arm."""
    root = cohort_root(data_root, cohort)
    if cohort == "gated":
        return {pid: d for pid, d, _ in gated_series(root)}
    return nongated_series(root)


def read_gated_reference(path):
    """{pid: reference score} from the CSV written by scripts/build_gated_reference.py. Patients with an
    ROI outside their slice stack (`out_of_range` > 0) are dropped: their score is wrong, not uncertain."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["out_of_range"]):
                continue
            out[int(r["patient_id"])] = float(r["total"])
    return out


def read_nongated_reference(path):
    """{pid: reference score} from the CSV written by scripts/export_nongated_reference.py."""
    with open(path) as f:
        return {int(r["patient_id"]): float(r["total"]) for r in csv.DictReader(f)}


def read_split(path):
    """Split CSV -> [(pid, split, agatston)] in file order. Columns: patient_id, split (train | test),
    agatston, cac_drs."""
    with open(path) as f:
        return [(int(r["patient_id"]), r["split"], float(r["agatston"])) for r in csv.DictReader(f)]


def write_split(path, rows):
    """rows: [(pid, split, agatston)] -> split CSV (sorted by patient id)."""
    from .agatston import cac_drs
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "split", "agatston", "cac_drs"])
        for pid, split, score in sorted(rows):
            w.writerow([pid, split, repr(float(score)), cac_drs(score)])
