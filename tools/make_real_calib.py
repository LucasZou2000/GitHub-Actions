"""Real-data calibration windows for the X5 PTQ rebuild.

Replaces the synthetic calibration set (calib/, accuracy INVALID) with windows
taken from the Parsed dataset -- one window per trial, sampled at the MIDDLE
valid endpoint so calibration windows differ from the eval-bundle windows
(bundle used linspace endpoints 0 and -1; this uses position 1 of 3).

Output: calib_real/raw_history_XXXX.bin, raw float32 [2,12,1,264] binaries
(post mirror/center/scale, 15 zero frames prefixed) -- the format hb_mapper
consumes. calib_ids.txt records which trial each window came from.

Usage:
    python tools/make_real_calib.py --dataset /path/to/Parsed
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRE = json.loads((ROOT / "models" / "preprocess.json").read_text())
HISTORY = int(PRE["history"])
GRAPH_LEN = int(PRE["graph_length"])
PREFIX = int(PRE["zero_prefix_frames"])
CENTER = np.asarray(PRE["center"], dtype=np.float32)
SCALE = np.asarray(PRE["scale"], dtype=np.float32)
MIRROR = np.asarray(PRE["mirror"], dtype=np.float32)[:, :, None]

FEATURES = [f"{segment}_imu_{side}_{kind}_{axis}"
            for side in ("l", "r") for segment in ("shank", "thigh")
            for kind in ("gyro", "accel") for axis in "xyz"]


def valid_window_ends(raw, times):
    bad_features = ~np.isfinite(raw).all(axis=(0, 1))
    bad_times = ~np.isfinite(times)
    intervals = np.diff(times)
    bad_intervals = ~np.isfinite(intervals) | (np.abs(intervals - .005) > 1e-5)
    ends = np.arange(HISTORY, len(times) + 1)
    rows = np.r_[0, np.cumsum(bad_features | bad_times)]
    gaps = np.r_[0, np.cumsum(bad_intervals)]
    valid = ((rows[ends] - rows[ends - HISTORY] == 0)
             & (gaps[ends - 1] - gaps[ends - HISTORY] == 0))
    return ends[valid]


def preprocess(raw):
    mirrored = raw.astype(np.float32) * MIRROR
    normalised = (mirrored - CENTER.reshape(1, 12, 1)) / SCALE.reshape(1, 12, 1)
    graph = np.zeros((2, 12, GRAPH_LEN), dtype=np.float32)
    graph[:, :, PREFIX:] = normalised
    return graph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "calib_real")
    args = ap.parse_args()
    files = [p for p in sorted(args.dataset.glob("BT*/*/*_exo.csv"))
             if p.name == f"{p.parent.parent.name}_{p.parent.name}_exo.csv"]
    if not files:
        sys.exit("No exact <subject>_<trial>_exo.csv sensor files found")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    ids_written = []
    for i, path in enumerate(files):
        relative = path.relative_to(args.dataset).as_posix()
        frame = pd.read_csv(path, usecols=["time"] + FEATURES)
        times = frame["time"].to_numpy(dtype=np.float64)
        raw = (frame[FEATURES].to_numpy(dtype=np.float32)
               .reshape(-1, 2, 12).transpose(1, 2, 0))
        valid_ends = valid_window_ends(raw, times)
        if not len(valid_ends):
            print(f"  SKIP (no valid window): {relative}", flush=True)
            continue
        # middle of 3 equally spaced positions: differs from the eval bundle
        # sampling (positions 0 and -1 of 2)
        pos = min(1, len(valid_ends) - 1)
        end = int(valid_ends[np.linspace(0, len(valid_ends) - 1, 3, dtype=int)[pos]])
        graph_in = preprocess(np.ascontiguousarray(raw[:, :, end - HISTORY:end]))
        (args.out / f"raw_history_{i:04d}.bin").write_bytes(graph_in.tobytes())
        ids_written.append(f"raw_history_{i:04d}.bin {relative}:{end}")
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f"progress {i + 1}/{len(files)}", flush=True)
    (args.out / "calib_ids.txt").write_text("\n".join(ids_written) + "\n")
    print(f"done: {len(ids_written)} calibration windows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
