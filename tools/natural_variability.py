#!/usr/bin/env python3
"""用生理自然变异给量化误差当标尺。

问题：INT8 的 RMSE 是 0.0197 Nm/kg，这个数算大还是算小？
答案：同一个人做同一个动作，不同窗口（不同步）之间的参考扭矩本身就在波动。
      如果量化误差远小于这个波动，它就不可能被控制侧或使用者感知。

做法：按 subject×activity 分组，组内要求 >=MIN_WIN 个窗口，
      统计参考扭矩跨窗口的标准差与半极差，取中位数作为自然变异水平。

输入：evidence/q8_eval/raw/reference_00{0..3}.npz
输出：evidence/q8_eval/natural_variability.csv + 终端摘要
"""
import collections
import os
import re

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "evidence", "q8_eval", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "evidence", "q8_eval")
JOINTS = ["LH", "LK", "RH", "RK"]
MIN_WIN = 25  # 少于此数的组合不参与统计（样本不足会把 std 算歪）


def main():
    refs, ids = [], []
    for i in range(4):
        r = np.load(os.path.join(RAW, f"reference_00{i}.npz"))
        refs.append(r["fp32_author"])
        ids.append(r["ids"])
    ref = np.concatenate(refs)
    ids = np.concatenate(ids)

    groups = collections.defaultdict(list)
    for k, s in enumerate(ids):
        m = re.match(r"^([^/]+)/(.+?)_\d+_\d+_", str(s))
        if m:
            groups[(m.group(1), m.group(2))].append(k)

    rows = []
    for (subj, act), ks in groups.items():
        if len(ks) < MIN_WIN:
            continue
        for j, joint in enumerate(JOINTS):
            v = ref[ks, j]
            rows.append(dict(subject=subj, activity=act, joint=joint,
                             n_windows=len(ks), std=float(v.std()),
                             half_range=float(np.ptp(v) / 2.0)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "natural_variability.csv"),
              index=False, float_format="%.6g")

    std_med = float(df["std"].median())
    rmse, mx = 0.0197, 0.1556  # accuracy.csv: ALL
    print(f"组合数 {len(df)}（每组 >= {MIN_WIN} 窗口）")
    print(f"窗口间 std     中位 {std_med:.4f}  "
          f"p10 {df['std'].quantile(.1):.4f}  p90 {df['std'].quantile(.9):.4f}")
    print(f"半极差         中位 {df['half_range'].median():.4f}")
    print(f"量化 RMSE {rmse:.4f} = 自然变异的 {100 * rmse / std_med:.1f}%")
    print(f"量化 max  {mx:.4f} = 自然变异的 {100 * mx / std_med:.1f}%")


if __name__ == "__main__":
    main()
