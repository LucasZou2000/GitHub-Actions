"""Synthetic calibration windows for the X5 PTQ spike.

NOT VALID FOR ACCURACY. These windows exist for one reason only: let
`hb_mapper makertbin` produce a .bin so that real board latency can be
measured. Any accuracy figure obtained from a model built with this set is
meaningless -- the int8 thresholds are derived from the distribution below,
not from recorded gait.

The graph input is the POST-mirror, POST-normalisation signal (mirror /
center / scale were moved to the CPU side), so windows are generated
directly in that space: per channel roughly N(0, 1), temporally smooth
(~1 Hz step fundamental plus 2nd/3rd harmonics), with a small fraction of
higher-amplitude windows so the thresholds are not under-estimated.

Usage:
    python tools/make_calib.py            # writes calib/*.npy + calib/synth_calib_100.npz
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRE = json.loads((ROOT / "models" / "preprocess.json").read_text())
HISTORY = int(PRE["history"])          # 249 real frames
PREFIX = int(PRE["zero_prefix_frames"])  # 15 explicit zero frames
GRAPH_LEN = int(PRE["graph_length"])   # 264
RATE = int(PRE["sample_hz"])           # 200 Hz
CH = len(PRE["center"])                # 12

N_WINDOWS = 100
N_HIGH_AMPLITUDE = 8                   # conservative tail widening
SEED = 20261118


def _leg(rng: np.random.Generator, phase: float, gain: float) -> np.ndarray:
    """One leg: [12, 249] in normalised units."""
    t = np.arange(HISTORY, dtype=np.float32) / RATE
    out = np.empty((CH, HISTORY), dtype=np.float32)
    for ch in range(CH):
        f = rng.uniform(0.80, 1.20)                     # step frequency, Hz
        a1, a2, a3 = 0.70, 0.35, 0.18
        p1, p2, p3 = rng.uniform(0, 2 * np.pi, 3)
        sig = a1 * np.sin(2 * np.pi * f * t + p1 + phase)
        sig += a2 * np.sin(2 * np.pi * 2 * f * t + p2 + phase * 1.7)
        sig += a3 * np.sin(2 * np.pi * 3 * f * t + p3 + phase * 0.6)
        noise = np.convolve(rng.standard_normal(HISTORY), np.ones(7) / 7, mode="same")
        out[ch] = gain * (sig + 1.6 * noise)
    return out


def window(rng: np.random.Generator, gain: float) -> np.ndarray:
    """[2, 12, 1, 264]: 15 zero frames then 249 real frames."""
    legs = np.stack([_leg(rng, 0.0, gain), _leg(rng, np.pi, gain)])   # [2,12,249]
    legs = np.clip(legs, -8.0, 8.0)
    zeros = np.zeros((2, CH, 1, PREFIX), dtype=np.float32)
    return np.concatenate([zeros, legs[:, :, None, :]], axis=3).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(SEED)
    calib_dir = ROOT / "calib"
    if not calib_dir.exists():
        calib_dir.mkdir()
    for old in calib_dir.glob("raw_history_*.npy"):
        old.unlink()

    windows = []
    for i in range(N_WINDOWS):
        gain = 1.8 if i < N_HIGH_AMPLITUDE else 1.0
        w = window(rng, gain)
        assert w.shape == (2, CH, 1, GRAPH_LEN), w.shape
        windows.append(w)
        np.save(calib_dir / f"raw_history_{i}.npy", w)

    stack = np.stack(windows)
    print(f"wrote {N_WINDOWS} npy files -> {calib_dir}")
    print(f"shape {stack.shape} dtype {stack.dtype}")
    print(f"normalised signal: mean {stack[..., PREFIX:].mean():+.3f} "
          f"std {stack[..., PREFIX:].std():.3f} "
          f"absmax {np.abs(stack).max():.2f}")


if __name__ == "__main__":
    main()

# NOTE: do NOT put any non-.npy file (including README) into calib/ --
# hb_mapper makertbin globs the whole directory, sorts it, and treats every
# file as one calibration sample ("First calibration picture name: README.md"
# -> "cannot reshape array of size 145 into shape (2,12,1,264)").
