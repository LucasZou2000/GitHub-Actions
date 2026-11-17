# GitHub-Actions — X5 BPU 可行性验证（离线 spike）

目的只有一件事：**用免费的 GitHub Actions 跑一次 `hb_mapper checker`**，回答"Nature TCN 的图能不能干净地落到 RDK X5 的 BPU（march `bayes-e`）上"。

- 不碰 StrideAssist 主仓、不改主线、不涉及硬件。
- 不训练、不重排权重、不做任何精度取舍；只改图的外部形态。
- 判定结果只回答"算子会不会被退回 CPU"，不构成任何性能或精度承诺。

## 为什么需要一个新的 ONNX

原始 `nature_latest_fp32.onnx` 是 rank-3 `[2,12,249]` + 1D 卷积，按 X5 官方算子表有几处会被判不合规：

| 改动 | 原因 |
|---|---|
| 输入改 rank-4 `[2,12,1,264]`，卷积改用 `(1,5)` 核 | X5 BPU 只加速 conv2d/conv3d，conv1d 只在 CPU 约束里 |
| 长度 249 → **264**（前面补 15 帧零） | 空洞卷积要求输入 W 能被 dilation 整除；249 时每层输入长度都是 `4d*k+1`，全部不整除 |
| mirror / center / scale 移出图外，由 CPU 做 | 原始输入张量把陀螺（每通道 scale 38~123）和加速度计（1.8~6.0）混在一个 per-tensor INT8 scale 里，加速度计通道会被压死 |
| 80→2 的线性头切掉，CPU 用 FP32 做 | 只有 160 MAC，放到 CPU 上成本为零，还能保住输出端精度 |

补 15 帧零是**严格等价**的：感受野正好 249 帧，补的零落在感受野之外，最后一帧输出不变。
等价性由 `tools/parity_check.py` 校验，实测 64 窗口 `max |diff| = 2.4e-07 Nm/kg`。

## 文件

```
models/nature_bpu.onnx      派生图（4D / 264 / 无归一化 / 无头），1.19 MB
models/preprocess.json      送数契约：mirror / center / scale / 头权重 / 零前缀帧数
tools/build_bpu_graph.py    从主仓模型重新生成上图（幂等）
tools/parity_check.py       与主仓原始图的数值等价校验（阈值 1e-5 Nm/kg）
.github/workflows/hb_checker.yml   Actions：拉 OE v1.2.8 镜像 → hb_mapper checker
```

注意：模型的第一维是**腿**，不是 batch。一次推理永远喂一条窗口的两条腿，输出 `[LH, LK, RH, RK]`。

## 本地复现（Mac）

```bash
python3 -m venv .venv && .venv/bin/pip install onnx onnxruntime numpy
.venv/bin/python tools/build_bpu_graph.py
.venv/bin/python tools/parity_check.py     # 期望 PARITY: PASS
```

## 在 Actions 上跑

仓库建议设为 **private**（2 000 分钟/月免费额度，本任务 10~30 分钟）。公有仓库虽然不限分钟，但会把模型权重公开。

Actions → `hb-checker` → Run workflow。它做四件事：

1. 删掉 runner 上预装的 ~25 GB 无关 SDK（OE 镜像很大）；
2. 从阿里云 OSS 下 `docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz`；
3. `docker load` 后在容器里跑 `hb_mapper checker --model-type onnx --march bayes-e --model models/nature_bpu.onnx`；
4. 把 `out/` 作为 artifact 上传，并在 job summary 里贴出算子放置摘要。

## 结果怎么读

看 artifact 里的 `checker.log`，关注三类信息：

- **有没有 `unsupported` / `not support` / `fallback`**：有就说明该算子会退回 CPU，图会被切成多个子图；
- **每个 Conv 被标成 BPU 还是 CPU**：目标是 11 个 Conv 全部 BPU；
- **子图切分数**：切得越碎，DMA + 同步的固定开销越大（本例 48 MMAC，算力本来就不是瓶颈，切碎才致命）。

三种结局：

| 结局 | 下一步 |
|---|---|
| 11 个 Conv 全 BPU、无切分 | 可以做 PTQ：需要真实校准窗口（200~1000 个 249 帧窗口），再跑 `makertbin` + `hb_verifier` |
| 少数层退回 CPU（例如末端 `W=16/1` 的极小卷积） | 仍可能可用，但要实测切分开销；或把末端几层留在 CPU FP32 |
| 空洞卷积被拒 / 大面积切分 | 放弃 BPU 路线，CPU（ONNX Runtime 或 streaming cache）已经够 10 Hz |

## 已知限制

- 14 GB 磁盘：镜像下载 + 加载后余量很小，workflow 里已经先清空间；失败时先看 `df -h` 那一步。
- 从国内 OSS 拉到境外 runner 可能很慢（5 GB 级）；超时就改用 registry 拉取（需要 `docker login`，不要把凭据写进仓库）。
- Actions runner 是临时 VM，产物必须走 artifact 带走。
- 这一步**只验证可编译性**，不验证精度。INT8 精度要另外用真实数据测：`models/preprocess.json` 里的归一化外移已经把误差从 ~32 降到 ~9 mNm/kg（合成数据仿真），真实窗口仍需复核。
