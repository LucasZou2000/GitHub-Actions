"""Exact-parity check: original packaged graph vs the derived BPU-shaped graph.

Reference : nature_latest_fp32.onnx fed with raw [2,12,249] (mirror/normalise inside).
Candidate : CPU preprocessing -> nature_bpu.onnx [2,12,1,264] -> FP32 head on the CPU.

Dim 0 of this model is the LEG axis, not a window batch: one window always carries two
legs and returns [LH, LK, RH, RK].  The only intended difference is float32 reassociation
inside the convolutions, so the tolerance is 1e-5 Nm/kg.
"""
from pathlib import Path
import json
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
REF = Path("/Users/lucas/Documents/Stride-Assist-main/Control/nature_tcn/models/nature_latest_fp32.onnx")
NEW = ROOT / "models/nature_bpu.onnx"
PRE = json.loads((ROOT / "models/preprocess.json").read_text())

PAD = PRE["zero_prefix_frames"]
LEN = PRE["graph_length"]
MIRROR = np.array(PRE["mirror"], np.float32)
CENTER = np.array(PRE["center"], np.float32).reshape(1, 12, 1)
SCALE = np.array(PRE["scale"], np.float32).reshape(1, 12, 1)


def synth(n, seed=7):
    """Walking-like windows following the checkpoint's own per-channel scales."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2, 12, 249), np.float32)
    t = np.arange(249) / 200.0
    for k in range(n):
        stride, phase = rng.uniform(0.8, 1.4), rng.uniform(0, 2 * np.pi)
        for leg in range(2):
            ph = phase + np.pi * leg
            for ch in range(12):
                base = CENTER[0, ch, 0] + SCALE[0, ch, 0] * (
                    1.1 * np.sin(2 * np.pi * stride * t + ph + 0.4 * ch))
                out[k, leg, ch] = base + rng.normal(0, 0.15 * SCALE[0, ch, 0], 249)
    return out


def to_graph_input(raw):
    """[2,12,249] raw -> [2,12,1,264] normalised with a zero prefix."""
    norm = (raw * MIRROR[:, :, None] - CENTER) / SCALE
    x = np.zeros((2, 12, 1, LEN), np.float32)
    x[:, :, 0, PAD:] = norm
    return x


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ref_s = ort.InferenceSession(str(REF), so, providers=["CPUExecutionProvider"])
    new_s = ort.InferenceSession(str(NEW), so, providers=["CPUExecutionProvider"])

    w = np.array(PRE["head_weight"], np.float32)   # [2,80]
    b = np.array(PRE["head_bias"], np.float32)     # [2]

    diffs, peaks = [], []
    for raw in synth(64):
        ref = ref_s.run(None, {"raw_history": raw})[0]                 # [4]
        feat = new_s.run(None, {"raw_history": to_graph_input(raw)})[0]  # [2,80,1,16]
        pred = (feat[:, :, 0, -1] @ w.T + b).reshape(-1)               # [4]
        diffs.append(np.abs(pred - ref).max())
        peaks.append(np.abs(ref).max())

    diffs = np.array(diffs)
    print(f"windows            : {len(diffs)}")
    print(f"output |ref| max   : {max(peaks):.3f} Nm/kg")
    print(f"max  |diff| Nm/kg  : {diffs.max():.3e}")
    print(f"mean |diff| Nm/kg  : {diffs.mean():.3e}")
    ok = diffs.max() < 1e-5
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
