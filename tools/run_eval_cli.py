"""Board-side INT8 accuracy evaluation via the vendor CLI.

Why not pyeasy_dnn: model.forward() through the python binding returns inflated
outputs (10x FP32) with a stale second batch slot, regardless of whether the
array is passed positionally, as a list, as a dict, or written into
model.inputs[0].buffer. `hrt_model_exec infer` on the same .bin matches the
FP32 graph to corr 0.9924, so the CLI is the trusted path.

Per window cost is dominated by process start (~1.25 s), so windows are spread
over N worker processes. Windows are split from graph_inputs.bin into temp .bin
files; CLI dumps a float32 [2,80,1,16] output per window.

Everything stays on the board: eval_results_cli/ keeps per-chunk outputs and
the summary.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

JOINTS = ("LH", "LK", "RH", "RK")
OUT_NAME = "model_infer_output_0__latest_blocks_blocks.4_Relu_2_output_0.bin"
SIGN_BAND = 0.05


def scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    err = pred.astype(np.float64) - ref.astype(np.float64)
    band = np.abs(ref) < SIGN_BAND
    out = {}
    for j, joint in enumerate(JOINTS):
        e = err[:, j]
        b = band[:, j]
        flips = int(np.sum(b & (np.sign(pred[:, j]) != np.sign(ref[:, j])) & (ref[:, j] != 0)))
        out[joint] = dict(count=int(len(e)), rmse=float(np.sqrt(np.mean(e ** 2))),
                          mae=float(np.mean(np.abs(e))), max_abs=float(np.max(np.abs(e))),
                          ref_rms=float(np.sqrt(np.mean(ref[:, j].astype(np.float64) ** 2))),
                          band_count=int(b.sum()), sign_flips=flips,
                          sign_flip_frac=float(flips / b.sum()) if b.sum() else 0.0)
    return out


def run_one(job):
    """One window through the vendor CLI (module level so it pickles)."""
    chunk, w, inp, outp, model = job
    dpath = outp + ".dir"
    os.makedirs(dpath, exist_ok=True)
    cmd = ["hrt_model_exec", "infer", "--model_file", model,
           "--input_file", inp, "--enable_dump", "true",
           "--hybrid_dequantize_process", "true", "--dump_path", dpath]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    src = pathlib.Path(dpath) / OUT_NAME
    os.replace(src, outp)
    shutil.rmtree(dpath, ignore_errors=True)
    return chunk, w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=pathlib.Path, default=pathlib.Path.home() / "data-transferrer")
    ap.add_argument("--model", type=pathlib.Path,
                    default=pathlib.Path.home() / "bpu_spike" / "nature_bpu_realcal.bin")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default=None, help="results subdir suffix, e.g. realcal")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    bundle = args.bundle.resolve()
    tag = args.tag or args.model.stem
    head = json.loads((bundle / "head.json").read_text())
    hw = np.asarray(head["head_weight"], dtype=np.float32)
    hb = np.asarray(head["head_bias"], dtype=np.float32)

    chunks = sorted(p for p in bundle.glob("chunk_*") if p.is_dir())
    if not chunks:
        raise SystemExit("no chunk_* dirs")
    results = bundle / "eval_results_cli" / tag
    work = pathlib.Path("/tmp/nature_eval") / tag
    (results / "dumps").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    refs, ids, preds = {}, {}, {}
    total = 0
    for cdir in chunks:
        ref = np.load(cdir / "reference.npz")
        data = np.fromfile(cdir / "graph_inputs.bin", dtype=np.float32).reshape(-1, 2, 12, 264)
        n = len(data)
        if args.limit:
            n = min(n, args.limit)
        refs[cdir.name] = dict(fp32_graph=ref["fp32_graph"][:n],
                               fp32_author=ref["fp32_author"][:n], ids=ref["ids"][:n])
        wdir = work / cdir.name
        wdir.mkdir(exist_ok=True)
        jobs = []
        for w in range(n):
            bin_path = wdir / f"w{w:04d}.bin"
            if not bin_path.exists():
                data[w].tofile(bin_path)
            jobs.append((cdir.name, w, str(bin_path),
                         str(results / "dumps" / f"{cdir.name}_w{w:04d}.bin"),
                         str(args.model.resolve())))

        t0 = time.perf_counter()
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            done = 0
            for chunk, w in pool.map(run_one, jobs, chunksize=8):
                done += 1
                if done % 200 == 0:
                    rate = (time.perf_counter() - t0) / done
                    print(f"{cdir.name}: {done}/{n}  {rate * 1000:.0f} ms/window", flush=True)
        elapsed = time.perf_counter() - t0
        print(f"{cdir.name}: {n} windows in {elapsed:.0f}s "
              f"({elapsed / n * 1000:.0f} ms/window, {args.workers} workers)", flush=True)

        out = np.empty((n, 4), dtype=np.float32)
        for w in range(n):
            feat = np.frombuffer((results / "dumps" / f"{cdir.name}_w{w:04d}.bin").read_bytes(),
                                 dtype=np.float32).reshape(2, 80, 16)[:, :, -1]
            out[w] = np.concatenate([feat[0] @ hw.T + hb, feat[1] @ hw.T + hb])
        preds[cdir.name] = out
        np.savez_compressed(results / f"board_outputs_{cdir.name}.npz",
                            int8_outputs=out, ids=refs[cdir.name]["ids"][:n])
        total += n

    pred_all = np.concatenate([preds[c.name] for c in chunks])
    graph_all = np.concatenate([refs[c.name]["fp32_graph"] for c in chunks])
    author_all = np.concatenate([refs[c.name]["fp32_author"] for c in chunks])
    summary = dict(windows=int(total), model=str(args.model), tag=tag,
                   sign_band_nm_kg=SIGN_BAND, workers=args.workers,
                   model_vs_graph_fp32=scores(pred_all, graph_all),
                   model_vs_author_fp32=scores(pred_all, author_all))
    (results / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
