"""Board-side evaluator: run the INT8 .bin on bundled real windows and score accuracy.

Runs ON THE RDK X5 (numpy + hobot_dnn only). Data stays on the board after the run.
For every window the final net moments are projected from the [2,80,1,16] feature
map with head.json (feature[:, :, 0, -1] @ head_weight.T + head_bias per leg).

Outputs (written next to the bundle, kept after the test):
  eval_results/board_outputs_chunk_XXX.npz   int8_outputs [W,4] + ids
  eval_results/eval_summary.json             per-chunk + aggregate accuracy metrics
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

JOINTS = ("LH", "LK", "RH", "RK")
SIGN_BAND = 0.05  # Nm/kg; sign-flip metric only inside |ref| < band


def scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    err = pred.astype(np.float64) - ref.astype(np.float64)
    band = np.abs(ref) < SIGN_BAND
    out = {}
    for j, joint in enumerate(JOINTS):
        e = err[:, j]
        b = band[:, j]
        flips = int(np.sum(b & (np.sign(pred[:, j]) != np.sign(ref[:, j]))
                           & (ref[:, j] != 0)))
        out[joint] = dict(
            count=len(e),
            rmse=float(np.sqrt(np.mean(e ** 2))),
            mae=float(np.mean(np.abs(e))),
            max_abs=float(np.max(np.abs(e))),
            ref_rms=float(np.sqrt(np.mean(ref[:, j].astype(np.float64) ** 2))),
            band_count=int(b.sum()),
            sign_flips=flips,
            sign_flip_frac=float(flips / b.sum()) if b.sum() else 0.0,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=pathlib.Path, default=pathlib.Path("."),
                    help="directory containing chunk_XXX/ and head.json")
    ap.add_argument("--model", type=pathlib.Path,
                    default=pathlib.Path.home() / "bpu_spike" / "nature_bpu_synthcal.bin")
    ap.add_argument("--save-features", type=int, default=0,
                    help="also store raw [2,80,16] feature maps for the first N windows")
    args = ap.parse_args()
    from hobot_dnn import pyeasy_dnn

    bundle = args.bundle.resolve()
    head = json.loads((bundle / "head.json").read_text())
    hw = np.asarray(head["head_weight"], dtype=np.float32)      # [2,80]
    hb = np.asarray(head["head_bias"], dtype=np.float32)        # [2]
    model = pyeasy_dnn.load(str(args.model.resolve()))[0]
    in_name = model.inputs[0].name  # informational only: forward() is positional

    # CRITICAL: model.forward({name: array}) silently ignores the data (the
    # binding logs "arg_obj is NOT a container") and every call returns the
    # same constant output. Positional forward(array) is deterministic and
    # input-sensitive -- verified by feeding zeros vs a real window and
    # checking the feature maps differ. Never go back to the dict form.
    def infer(graph_in: np.ndarray) -> np.ndarray:
        outs = model.forward(graph_in)
        return np.frombuffer(outs[0].buffer, dtype=np.float32).reshape(2, 80, 16)

    results_dir = bundle / "eval_results"
    results_dir.mkdir(exist_ok=True)
    chunks = sorted(p for p in bundle.glob("chunk_*") if p.is_dir())
    if not chunks:
        raise SystemExit(f"no chunk_XXX directories under {bundle}")
    print(f"model={args.model.name} chunks={len(chunks)} -> {results_dir}", flush=True)

    aggregate = {"graph_vs_int8": {}, "author_vs_int8": {}}
    total = 0
    t0 = time.perf_counter()
    saved_features = 0
    for ci, cdir in enumerate(chunks):
        ref = np.load(cdir / "reference.npz")
        ids = ref["ids"]
        n = len(ids)
        data = np.fromfile(cdir / "graph_inputs.bin", dtype=np.float32).reshape(n, 2, 12, 264)
        int8_out = np.empty((n, 4), dtype=np.float32)
        feat_keep = []
        for w in range(n):
            graph_in = np.ascontiguousarray(data[w, :, :, None, :])  # [2,12,1,264]
            feat_full = infer(graph_in)
            feat = feat_full[:, :, -1]
            left = feat[0] @ hw.T + hb
            right = feat[1] @ hw.T + hb
            int8_out[w] = np.concatenate([left, right])
            if saved_features < args.save_features:
                feat_keep.append(feat_full.copy())
                saved_features += 1
            if (w + 1) % 200 == 0 or w + 1 == n:
                rate = (time.perf_counter() - t0) / (total + w + 1)
                print(f"{cdir.name}: {w + 1}/{n}  ({rate * 1000:.2f} ms/window avg)", flush=True)
        np.savez_compressed(results_dir / f"board_outputs_{cdir.name}.npz",
                            int8_outputs=int8_out, ids=ids)
        if feat_keep:
            np.savez_compressed(results_dir / f"board_features_{cdir.name}.npz",
                                features=np.stack(feat_keep))
        s_graph = scores(int8_out, ref["fp32_graph"])
        s_author = scores(int8_out, ref["fp32_author"])
        json.dump(dict(chunk=cdir.name, model_vs_graph_fp32=s_graph,
                       model_vs_author_fp32=s_author),
                  (results_dir / f"eval_chunk_{ci:03d}.json").open("w"), indent=2)
        total += n

    # weighted aggregate: rmse over all windows, recomputed from stored outputs
    def pooled(which: str) -> dict:
        preds, refs = [], []
        for cdir in chunks:
            ref = np.load(cdir / "reference.npz")
            refs.append(ref["fp32_graph" if which == "graph" else "fp32_author"])
            preds.append(np.load(results_dir / f"board_outputs_{cdir.name}.npz")["int8_outputs"])
        return scores(np.concatenate(preds), np.concatenate(refs))

    summary = dict(windows=total, elapsed_s=round(time.perf_counter() - t0, 1),
                   model=str(args.model), sign_band_nm_kg=SIGN_BAND,
                   pooled_model_vs_graph_fp32=pooled("graph"),
                   pooled_model_vs_author_fp32=pooled("author"))
    (results_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
