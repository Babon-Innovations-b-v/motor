"""A demo clip's measured vertical ground reaction force per foot, for compose_clip.py to draw beside MOTOR's: the
trial's own force plates from LabValidation, read, aligned and filtered as SCORE's LabValidation scorer reads them
(`tools.labval`: the run shifted onto the lab's clock by its angles' best match to the lab's inverse kinematics, the
plates low-passed at the activity's OpenCap cutoff), resampled at each of the run's frames and over the subject's
weight (kg times 9.81), % body weight; nan where the plates were not recording. Run in SCORE's Python.

    python plates.py <SCORE checkout> <LabValidation folder> <trial folder> <out.npz>"""
import json
import pathlib
import sys

import numpy as np


def plate_curves(labval, folder):
    """{side: % body weight at each of the run's frames} from the trial's force plates, the scorer's alignment."""
    from tools.labval import _files, _grf, _run, _score, activity_of, subject_of
    record = json.loads((folder / "record.json").read_text())
    subject_name, trial, _ = subject_of(record["name"])
    subject = labval / subject_name
    entry, arrays, rate = _run.read(folder)
    weight = _files.metadata(_files.found(subject, "metadata", trial))["kg"] * _files.G
    times = float(record["start"]) + np.arange(len(entry["frames"])) / rate
    ik_times, ik_columns = _files.ik(_files.found(subject, "ik", trial))
    at = times + _score.shift(times, _run.angles(entry, arrays, 0.0), ik_times, ik_columns)
    cutoff = _grf.cutoff_of(activity_of(trial), rate)
    plate_times, plates = _files.forces(_files.found(subject, "forces", trial))
    return {side: _score.resampled(plate_times, _grf.filtered(plate_times, plates[side]["force"], cutoff), at)[:, 1]
            / weight * 100 for side in ("right", "left")}


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve()))
    found = plate_curves(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    np.savez(sys.argv[4], **found)
    print(f"plates: {sys.argv[4]} ({len(found['right'])} frames, peaks right {np.nanmax(found['right']):.0f} "
          f"left {np.nanmax(found['left']):.0f} % body weight)")
