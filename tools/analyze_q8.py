#!/usr/bin/env python3
"""Q8 量化模型精度评估（相对误差百分比 / 分位 / 分档 / 偏差）

输入：evidence/q8_eval/raw/ 下 4 个 chunk 的 reference.npz（FP32 双参考）
      与 board_outputs_chunk_*.npz（板上 INT8 输出）
输出：accuracy.csv（逐关节 + 汇总）、accuracy_bins.csv（按扭矩幅值分档）
"""
import json
import os

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "evidence", "q8_eval", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "evidence", "q8_eval")
JOINTS = ["LH", "LK", "RH", "RK"]
BAND = 0.05  # Nm/kg，小于此值方向无物理意义


def load_all():
    ys, rs, ids = [], [], []
    for i in range(4):
        b = np.load(os.path.join(RAW, f"board_outputs_chunk_00{i}.npz"))
        r = np.load(os.path.join(RAW, f"reference_00{i}.npz"))
        ys.append(b["int8_outputs"])
        rs.append(r["fp32_author"])
        ids.append(b["ids"])
    return np.concatenate(ys), np.concatenate(rs), np.concatenate(ids)


def metrics(ref, pred):
    err = pred - ref
    ae = np.abs(err)
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sst = float(np.sum((ref - ref.mean()) ** 2))
    sse = float(np.sum(err ** 2))
    corr = float(np.corrcoef(ref, pred)[0, 1]) if ref.std() > 0 else float("nan")
    band = np.abs(ref) >= BAND
    big = np.abs(ref) >= 0.10
    high = np.abs(ref) >= 0.5
    dot = float(np.sum(ref * pred))
    cos = dot / (float(np.linalg.norm(ref)) * float(np.linalg.norm(pred)) + 1e-12)
    snr_db = 10.0 * np.log10(float(np.sum(ref ** 2)) / max(float(np.sum(err ** 2)), 1e-20))
    enob = np.log2(ref_rms / rmse) if rmse > 0 else float("nan")
    d = dict(
        n=int(ref.size),
        ref_rms=ref_rms,
        ref_abs_max=float(np.abs(ref).max()),
        rmse=rmse,
        mae=float(ae.mean()),
        abs_p95=float(np.percentile(ae, 95)),
        abs_p99=float(np.percentile(ae, 99)),
        abs_max=float(ae.max()),
        rel_rmse_pct=100.0 * rmse / ref_rms,
        cos=cos,
        snr_db=float(snr_db),
        enob_bits=float(enob),
        rel_p99_pct=100.0 * float(np.percentile(ae, 99)) / ref_rms,
        r2=1.0 - sse / sst,
        corr=corr,
        bias=float(err.mean()),
        band_n=int(band.sum()),
        band_sign_agree_pct=100.0 * float(np.mean(np.sign(pred[band]) == np.sign(ref[band]))),
        big_n=int(big.sum()),
        big_sign_agree_pct=100.0 * float(np.mean(np.sign(pred[big]) == np.sign(ref[big]))),
        high_n=int(high.sum()),
        high_rel_err_pct=100.0 * float(np.mean(ae[high] / np.abs(ref[high]))) if high.any() else float("nan"),
    )
    return d


def main():
    y, r, ids = load_all()
    print(f"windows={len(y)}  outputs={y.shape}  unique ids={len(set(ids))}")

    rows = []
    for j, name in enumerate(JOINTS):
        d = metrics(r[:, j].astype(np.float64), y[:, j].astype(np.float64))
        d = {"joint": name, **d}
        rows.append(d)
    pooled = metrics(r.ravel().astype(np.float64), y.ravel().astype(np.float64))
    rows.append({"joint": "ALL", **pooled})
    acc = pd.DataFrame(rows)
    acc.to_csv(os.path.join(OUT, "accuracy.csv"), index=False, float_format="%.6g")

    # 按活动类型
    import re as _re
    acts = []
    for s_ in ids:
        m = _re.match(r"^[^/]+/(.+?)_\d+_\d+_", str(s_))
        acts.append(m.group(1) if m else "other")
    acts = np.array(acts)
    arows = []
    for act in sorted(set(acts)):
        m = acts == act
        if m.sum() < 20:
            continue
        d = metrics(r[m].ravel().astype(np.float64), y[m].ravel().astype(np.float64))
        arows.append({"activity": act, "n_windows": int(m.sum()),
                      "rmse": d["rmse"], "rel_rmse_pct": d["rel_rmse_pct"],
                      "abs_p99": d["abs_p99"], "abs_max": d["abs_max"],
                      "big_sign_agree_pct": d["big_sign_agree_pct"]})
    adf = pd.DataFrame(arows).sort_values("n_windows", ascending=False)
    adf.to_csv(os.path.join(OUT, "accuracy_activity.csv"), index=False, float_format="%.6g")
    print("\n=== per-activity (n>=20) ===")
    print(adf.head(12).to_string(index=False, float_format="%.4f"))

    # 按扭矩幅值分档
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0, np.inf]
    labels = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5", "0.5-1.0", ">=1.0"]
    brows = []
    for j, name in enumerate(JOINTS + ["ALL"]):
        if name == "ALL":
            ref, pred = r.ravel().astype(np.float64), y.ravel().astype(np.float64)
        else:
            ref, pred = r[:, j].astype(np.float64), y[:, j].astype(np.float64)
        idx = np.digitize(np.abs(ref), bins[1:-1])
        for k, lab in enumerate(labels):
            m = idx == k
            if not m.any():
                continue
            err = pred[m] - ref[m]
            brows.append(dict(
                joint=name, bin=lab, n=int(m.sum()),
                ref_rms=float(np.sqrt(np.mean(ref[m] ** 2))),
                rmse=float(np.sqrt(np.mean(err ** 2))),
                rel_rmse_pct=100.0 * float(np.sqrt(np.mean(err ** 2))) /
                             max(float(np.sqrt(np.mean(ref[m] ** 2))), 1e-9),
                abs_p99=float(np.percentile(np.abs(err), 99)),
                abs_max=float(np.abs(err).max()),
                sign_agree_pct=100.0 * float(np.mean(np.sign(pred[m]) == np.sign(ref[m]))),
            ))
    pd.DataFrame(brows).to_csv(os.path.join(OUT, "accuracy_bins.csv"), index=False,
                               float_format="%.6g")

    # FP32 板上（C++ onnxruntime） vs author FP32 参考
    refp = np.load(os.path.join(RAW, "ref_author.npz"))
    ort = np.fromfile(os.path.join(RAW, "ort_out_128.bin"), dtype=np.float32).reshape(-1, 4)
    author = refp["author"].astype(np.float64)
    n = min(len(ort), len(author))
    d = np.abs(ort[:n].astype(np.float64) - author[:n])
    fp32 = dict(n=int(n), max_abs=float(d.max()), rmse=float(np.sqrt(np.mean(d ** 2))),
                rel_pct=100.0 * float(d.max()) / float(np.abs(author[:n]).max()))
    print("\nFP32 on-board (onnxruntime CPU, C++) vs author reference:",
          json.dumps(fp32, indent=1))

    # C++ DUMP_ALL 与 run_eval_cli（python CLI）一致性
    b = np.load(os.path.join(RAW, "board_outputs_chunk_000.npz"))["int8_outputs"]
    cpp = np.fromfile(os.path.join(RAW, "bpu_out_a.bin"), dtype=np.float32).reshape(-1, 4)
    m = min(len(b), len(cpp))
    dd = np.abs(b[:m].astype(np.float64) - cpp[:m].astype(np.float64))
    print(f"C++ hbDNN dump vs CLI dump: n={m} max_abs={dd.max():.3e}")

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump({"fp32_board_vs_author": fp32,
                   "cpp_vs_cli": {"n": int(m), "max_abs": float(dd.max())},
                   "windows": int(len(y))}, f, indent=1)

    print("\n=== per-joint ===")
    print(acc[["joint", "n", "rmse", "rel_rmse_pct", "abs_p99", "abs_max",
               "r2", "cos", "snr_db", "enob_bits", "bias"]].to_string(index=False,
                                                                     float_format="%.4f"))


if __name__ == "__main__":
    main()
