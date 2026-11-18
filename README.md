# GitHub-Actions — X5 BPU 可行性验证（离线 spike）

回答一个问题：**Nature TCN 的图能不能干净地落到 RDK X5 的 BPU（march `bayes-e`）上。**

- 不碰 StrideAssist 主仓、不动主线代码、不涉及硬件。
- 不训练、不重排权重；只改图的外部形态。
- 结论只覆盖"能不能编译 / 算子会不会退回 CPU / 静态延迟量级"，**不构成精度承诺**。

## 结论（已实测，2026-11-18）

工具链：`hb_mapper 1.24.3` / `hbdk 3.49.15` / `horizon_nn 1.1.0`，OE v1.2.8，`--march bayes-e`。

| 项目 | 结果 |
|---|---|
| **算子放置** | **25 / 25 节点全部 BPU，零 CPU 回退** |
| **子图切分** | **1 个**（`main_graph_subgraph_0`），没有碎片化 |
| 量化类型 | 全 int8；末层 `Relu_55` 为 `HzLut2Layer` int16（输出层自动提精度） |
| checker 静态延迟 | **1967.7 µs（≈2.0 ms）**，hbdk-cc 评估 `FPS=1016.4`，DDR 3.59 MB/次 |
| **makertbin 编译后延迟** | **361.6 µs**，`FPS=5531`，DDR 313 408 B ≈ 0.30 MB/次 |
| 量化逐层余弦相似度 | 全部 ≥ 0.9998；图输出 `0.999737`（合成校准，仅供流程参考，**不是精度结论**） |
| **可部署产物** | **`nature_bpu_synthcal.bin`（526 KB）已生成**，`Convert to runtime bin file successfully!` |

两个延迟数字的差异值得说明：checker 的 2.0 ms 是**编译前**按节点逐个估的保守值；makertbin 出来的 361.6 µs 是**编译后**针对实际定点图的评估——优化后的图比逐节点估计好 5 倍多。真实数字仍然以板上 `hrt_model_exec perf` 为准。

**必须先过的一关：opset。** X5 的 ONNX 前端**最高只支持 opset 11**，原始导出是 opset 17 / IR 8，直接报：

```
ERROR *** ERROR-OCCUR-DURING {horizon_nn.build_onnx} ***,
error message: The opset version of the model is 17, the maximum supported version is 11.
```

`hb_mapper` 用 `sys.exit(-1)` 退出，所以看到的是 **exit 255**。降为 opset 11 / IR 7 后一次通过，且与 opset 17 版本 **逐位一致**（`max diff = 0.0`）。

**延迟这一项推翻了我之前的算术估计。** 按 48 MMAC ÷ 5 TMAC/s 推出的 ~10 µs 是错的，实测 **2 ms**，差 200 倍。原因不是算力，而是：

- 25 个算子每个都是一条独立 BPU 指令，各自要 DDR 往返；张量太小（末端 `[2,80,1,16]` = 10 KB），固定开销完全主导；
- 卷积是 `1×5` 深度极浅的 1D 结构，`W=1`，TAE 阵列填不满（末端 `W=16`）。

也就是说"BPU 更适合大矩阵"那个口径在**利用率**上是对的，但它不构成"不能用"——2 ms 在 10 Hz 下只占 **2%** 占空比。

占空比推算：**10 Hz → 2.0%**；**200 Hz → 39.4%**（且无法与自身流水，除非多核 / 多图并行）。DDR 侧 10 Hz 只有 37.6 MB/s，无压力。

## 还没验证的两件事

1. **精度**。`.bin` 的校准集是**合成的**（`tools/make_calib.py`，固定种子），量化阈值来自合成分布。逐层 cosine 0.9998 只说明量化数值上收敛，**不是真实步态上的精度结论**。必须用真实 `*_exo.csv` 窗口重新校准再评估。
2. **板上实测**。361.6 µs 是 x86 侧编译评估，实板通常更高（DDR 竞争、调度）。要 `hrt_model_exec perf` 才算数。

## 为什么需要一个新的 ONNX

原始 `nature_latest_fp32.onnx` 是 rank-3 `[2,12,249]` + 1D 卷积，按 X5 算子表有几处会不合规：

| 改动 | 原因 |
|---|---|
| 输入改 rank-4 `[2,12,1,264]`，卷积改用 `(1,5)` 核 | X5 BPU 只加速 conv2d/conv3d，conv1d 只在 CPU 约束里 |
| 长度 249 → **264**（前置 15 帧零） | 空洞卷积要求输入 W 能被 dilation 整除；249 时每层输入长度都是 `4d·k+1`，全部不整除 |
| mirror / center / scale 移出图外，由 CPU 做 | 原输入把陀螺（每通道 scale 38~123）和加速度计（1.8~6.0）混进同一个 per-tensor INT8 scale，加速度计通道会被压死 |
| 80→2 线性头切掉，CPU 用 FP32 做 | 只有 160 MAC，放 CPU 成本为零，还能保住输出端精度 |
| 版本戳降为 **opset 11 / IR 7** | 前端硬限制，见上 |

补 15 帧零是**严格等价**的：感受野正好 249 帧，补的零在感受野之外。等价性由 `tools/parity_check.py` 校验，实测 64 窗口 `max |diff| = 2.4e-07 Nm/kg`；opset 降级再用 ORT 复验，与 opset 17 **逐位相同**。

## 文件

```
models/nature_bpu.onnx           派生图（4D / 264 / 无归一化 / 无头），opset 17，1.19 MB
models/nature_bpu_opset11.onnx   同一张图，opset 11 / IR 7 —— X5 前端只吃这个
models/preprocess.json           送数契约：mirror / center / scale / 头权重 / 零前缀帧数
tools/build_bpu_graph.py         从主仓模型重新生成上面两张图（幂等）+ preprocess.json
tools/parity_check.py            与主仓原始图的数值等价校验（阈值 1e-5 Nm/kg）
tools/make_calib.py              合成校准集生成器（固定种子，仅供跑通编译，精度无效）
config.makertbin.yaml            makertbin A 配置（正式）；config.b.yaml 为最小回退配置
.github/workflows/hb_checker.yml    Actions：hb_mapper checker → 日志回推 logs 分支
.github/workflows/hb_makertbin.yml  Actions：合成校准 → makertbin → 产物回推 logs-makertbin
evidence/                        已实测运行的完整日志与产物存档
```

第一维是**腿**，不是 batch。一次推理喂一条窗口的两条腿，输出 `[LH, LK, RH, RK]`。

## 工具链踩坑记录（v1.2.8 实测）

| 现象 | 原因 / 修法 |
|---|---|
| opset 17 直接拒收 | 前端最高 opset 11 / IR 7；图内算子在 11 中语义相同，直接改版本戳即可，无需 version_converter |
| `Wrong key 'cal_sample_num'` | v1.2.8 的 schema 无此键；工具会读 `cal_data_dir` 下全部样本文件 |
| `Input_layout_rt missing` | `input_type_rt: featuremap` 时 schema 强制要求 `input_layout_rt` |
| config 放 `/tmp` 后找不到 `onnx_model` | **配置内相对路径按配置文件所在目录解析**，不是按 cwd |
| `First calibration picture name: README.md` | 工具对 `cal_data_dir` **全量 glob、不过滤扩展名**；目录里不能放任何非样本文件 |
| `size 6368 vs 6336` | `cal_data_type: float32` 按**裸二进制**读样本，不做 npy 解析——128 B 的 npy 头会混进数据；样本必须 `tofile()` 写裸 float32 |
| `march=x5` 无效 | makertbin 只认 `bernoulli2 / bayes / bayes-e` |
| CI 绿了但实际失败 | `continue-on-error` 掩盖 exit 255；**必须查 check-run annotations** |

## 本地复现（Mac）

```bash
python3 -m venv .venv && .venv/bin/pip install onnx onnxruntime numpy
.venv/bin/python tools/build_bpu_graph.py
.venv/bin/python tools/parity_check.py     # 期望 PARITY: PASS
```

## 在 Actions 上跑

仓库保持 **public**（分钟数不限；内容本身就是公开论文模型重新导出的形态，无隐私顾虑）。

Actions → `hb-checker` → Run workflow。它做：

1. 删掉 runner 上预装的 ~25 GB 无关 SDK；
2. 取 `docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz`（**1.69 GiB**，`actions/cache` 命中时跳过下载）；
3. `docker load` 后在容器里依次尝试 `opset17 → opset11 → opset11 + --input-shape`，每次打 `CASE` 标记；
4. 把完整控制台日志推到 **`logs` 分支**（`codeload` 可直接取，不必下载 artifact），同时上传 artifact。

单次运行约 5 分钟（冷缓存），热缓存约 1 分钟。

## 结果怎么读

`logs` 分支的 `full_console.log` 里，看三件事：

- **`ERROR-OCCUR-DURING`**：最典型的是 opset 超限，直接降版本；
- **算子表里 `ON` 列**：应为全 `BPU`，出现 `CPU` 说明该算子退回；
- **`Subgraph` 列**：全是 `id(0)` 表示单子图，出现 `id(1)`/`id(2)` 说明被切碎，DMA 与同步开销会吃掉收益。

## 下一步

| 目标 | 需要什么 |
|---|---|
| ✅ 拿到可部署 `.bin` | **已完成**：`evidence/makertbin_run44217406326/nature_bpu_synthcal.bin`（526 KB，合成校准） |
| 板上实测延迟 | 把 `.bin` 拷到 X5，`hrt_model_exec perf --model nature_bpu_synthcal.bin` |
| 判定 INT8 精度 | **真实校准窗口**（200~1000 个 249 帧窗口，来自 `*_exo.csv`）重跑 makertbin + 逐关节 RMSE 对比 |
| 决定是否切主线 | 上面两项都有数之后。当前主线的 10 Hz CPU 路径本身已达标，BPU 的收益是"回收约 1.6 个 A55 核 + 留余量"，不是"达标手段" |

## 已知限制

- `nature_bpu_synthcal.bin` 的校准集是合成的，**只可用于延迟测量，精度结论无效，不得流入主仓**。
- 真实数据校准的 `.bin` 命名必须与 synthcal 区分开。
- Actions runner 是临时 VM，产物必须走 artifact 或 `logs` 分支带走。
