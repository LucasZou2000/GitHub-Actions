"""Build a real-data evaluation bundle for the X5 board from the Parsed dataset.

Every window is an exact author input [2,12,249] taken from <subject>_<trial>_exo.csv
sensor columns (raw author sensor frames, BEFORE left mirroring). Windows with NaN
rows or non-200 Hz gaps are excluded -- never filled or bridged.

Bundle layout (--out):
  chunk_000/graph_inputs.bin   raw float32, W x (2*12*264): post-mirror, post
                               (center/scale) normalisation, 15 zero frames prefixed
  chunk_000/reference.npz      fp32_author [W,4]  -- nature_latest_fp32.onnx on raw windows
                               fp32_graph  [W,4]  -- nature_bpu_opset11.onnx on graph inputs,
                                                     head applied (isolate INT8 quantization)
                               ids         [W]    -- "<subject>/<trial>:<end>"
  head.json                    2x80 head weights + bias for the board-side projection
  run_eval.py                  board runner (numpy + hobot_dnn only, no other deps)

FP32 references are computed with 2 ORT threads to stay light on the user's Mac.
The board model keeps SYNTHETIC calibration thresholds; this bundle measures what
that INT8 model actually does on recorded gait.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRE = json.loads((ROOT / "models" / "preprocess.json").read_text())
HISTORY = int(PRE["history"])            # 249
GRAPH_LEN = int(PRE["graph_length"])     # 264
PREFIX = int(PRE["zero_prefix_frames"])  # 15
CENTER = np.asarray(PRE["center"], dtype=np.float32)
SCALE = np.asarray(PRE["scale"], dtype=np.float32)
MIRROR = np.asarray(PRE["mirror"], dtype=np.float32)

FEATURES = [f"{segment}_imu_{side}_{kind}_{axis}"
            for side in ("l", "r") for segment in ("shank", "thigh")
            for kind in ("gyro", "accel") for axis in "xyz"]

FP32_AUTHOR = pathlib.Path("/Users/lucas/Documents/Stride-Assist-main/"
                           "Control/nature_tcn/models/nature_latest_fp32.onnx")
FP32_GRAPH = ROOT / "models" / "nature_bpu_opset11.onnx"


def sha256_file(path: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_window_ends(raw: np.ndarray, times: np.ndarray):
    """Endpoints of complete finite 200 Hz windows; invalid samples are never filled."""
    bad_features = ~np.isfinite(raw).all(axis=(0, 1))
    bad_times = ~np.isfinite(times)
    intervals = np.diff(times)
    bad_intervals = ~np.isfinite(intervals) | (np.abs(intervals - .005) > 1e-5)
    ends = np.arange(HISTORY, len(times) + 1)
    rows = np.r_[0, np.cumsum(bad_features | bad_times)]
    gaps = np.r_[0, np.cumsum(bad_intervals)]
    valid = ((rows[ends] - rows[ends - HISTORY] == 0)
             & (gaps[ends - 1] - gaps[ends - HISTORY] == 0))
    excluded = int((~valid).sum())
    return ends[valid], excluded


def preprocess(raw: np.ndarray) -> np.ndarray:
    """raw [2,12,249] author frames -> graph input [2,12,264] (post mirror/center/scale + zero prefix)."""
    mirrored = raw.astype(np.float32) * MIRROR[:, :, None]
    normalised = (mirrored - CENTER.reshape(1, 12, 1)) / SCALE.reshape(1, 12, 1)
    graph = np.zeros((2, 12, GRAPH_LEN), dtype=np.float32)
    graph[:, :, PREFIX:] = normalised
    return graph


def dense_trial_pick(files, keyword: str):
    choices = [p for p in files if keyword in p.parent.name]
    preferred = [p for p in choices if p.parent.parent.name == "BT18"]
    return (preferred or choices)[0] if choices else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--windows-per-trial", type=int, default=2)
    ap.add_argument("--dense-windows", type=int, default=150,
                    help="extra dense windows per dense trial (normal_walk/squat/jump)")
    ap.add_argument("--chunk", type=int, default=600, help="windows per chunk_XXX dir")
    ap.add_argument("--limit-trials", type=int, default=0, help="0 = all exact sensor files")
    args = ap.parse_args()

    if not FP32_AUTHOR.exists():
        sys.exit(f"missing {FP32_AUTHOR}")
    files = [p for p in sorted(args.dataset.glob("BT*/*/*_exo.csv"))
             if p.name == f"{p.parent.parent.name}_{p.parent.name}_exo.csv"]
    if args.limit_trials:
        files = files[:args.limit_trials]
    if not files:
        sys.exit("No exact <subject>_<trial>_exo.csv sensor files found")

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 2
    sess_author = ort.InferenceSession(str(FP32_AUTHOR), sess_options=options,
                                       providers=["CPUExecutionProvider"])
    sess_graph = ort.InferenceSession(str(FP32_GRAPH), sess_options=options,
                                      providers=["CPUExecutionProvider"])

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    head = dict(head_weight=PRE["head_weight"], head_bias=PRE["head_bias"],
                output_order=PRE["output_order"], output_units=PRE["output_units"],
                graph_output=PRE["graph_output"])
    (out / "head.json").write_text(json.dumps(head, indent=2))
    shutil.copy2(ROOT / "tools" / "run_eval.py", out / "run_eval.py")

    dense = {t.relative_to(args.dataset).as_posix(): args.dense_windows
             for t in (dense_trial_pick(files, k) for k in ("normal_walk", "squat", "jump"))
             if t is not None}

    stats = dict(trials=len(files), dense=dense, windows=0, excluded_endpoints=0,
                 fp32_author_sha256=sha256_file(FP32_AUTHOR),
                 fp32_graph_sha256=sha256_file(FP32_GRAPH))
    manifest_path = out / "manifest.json"

    def flush(index, w_in, ids_in, a_in, g_in):
        if not ids_in:
            return
        cdir = out / f"chunk_{index:03d}"
        cdir.mkdir()
        np.stack(w_in).tofile(cdir / "graph_inputs.bin")  # raw float32, no header
        np.savez_compressed(cdir / "reference.npz",
                            fp32_author=np.stack(a_in).astype(np.float32),
                            fp32_graph=np.stack(g_in).astype(np.float32),
                            ids=np.asarray(ids_in))
        print(f"  flushed {cdir.name}: {len(ids_in)} windows", flush=True)

    pending_w, pending_ids, pending_a, pending_g = [], [], [], []
    chunk_index = 0
    for ti, path in enumerate(files):
        relative = path.relative_to(args.dataset).as_posix()
        frame = pd.read_csv(path, usecols=["time"] + FEATURES)
        times = frame["time"].to_numpy(dtype=np.float64)
        raw = (frame[FEATURES].to_numpy(dtype=np.float32)
               .reshape(-1, 2, 12).transpose(1, 2, 0))
        valid_ends, excluded = valid_window_ends(raw, times)
        stats["excluded_endpoints"] += excluded
        if not len(valid_ends):
            print(f"  SKIP (no valid window): {relative}", flush=True)
            continue
        positions = np.linspace(0, len(valid_ends) - 1, args.windows_per_trial, dtype=int)
        ends = list(valid_ends[positions])
        if relative in dense:
            k = min(dense[relative], len(valid_ends))
            ends.extend(valid_ends[:k])
        for end in sorted(ends):
            raw_w = np.ascontiguousarray(raw[:, :, end - HISTORY:end])
            graph_in = preprocess(raw_w)
            a = sess_author.run(None, {"raw_history": raw_w})[0]
            g_feat = sess_graph.run(None, {"raw_history": graph_in[:, :, None, :]})[0]
            # graph output [2,80,1,16] -> [2,80] per leg -> head -> [2,2]
            feats = g_feat[:, :, 0, -1]
            g = np.concatenate([feats[0] @ np.asarray(PRE["head_weight"]).T + PRE["head_bias"],
                                feats[1] @ np.asarray(PRE["head_weight"]).T + PRE["head_bias"]])
            pending_w.append(graph_in)
            pending_ids.append(f"{relative}:{end}")
            pending_a.append(a.ravel())
            pending_g.append(g.ravel())
            stats["windows"] += 1
        if len(pending_ids) >= args.chunk:
            flush(chunk_index, pending_w, pending_ids, pending_a, pending_g)
            chunk_index += 1
            pending_w, pending_ids, pending_a, pending_g = [], [], [], []
        if (ti + 1) % 25 == 0 or ti + 1 == len(files):
            print(f"progress {ti + 1}/{len(files)} trials, {stats['windows']} windows", flush=True)
    flush(chunk_index, pending_w, pending_ids, pending_a, pending_g)

    manifest_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "dense"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
