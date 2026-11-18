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
| **X5 板上实测** | **平均 1.39–1.65 ms，FPS 598–711**（单线程、core 0、2000 帧×2 轮；min 0.67 ms / max 2.68 ms） |
| 量化逐层余弦相似度 | 全部 ≥ 0.9998；图输出 `0.999737`（合成校准，仅供流程参考，**不是精度结论**） |
| **可部署产物** | **`nature_bpu_synthcal.bin`（526 KB）已生成**，`Convert to runtime bin file successfully!` |

两个延迟数字的差异值得说明：checker 的 2.0 ms 是**编译前**按节点逐个估的保守值；makertbin 出来的 361.6 µs 是**编译后**针对实际定点图的评估——但**板上实测落在两者之外：1.4–1.7 ms，更接近 checker 的估计**。x86 侧编译评估不含 DDR 传输与调度开销，不能当板上性能用。

板上占空比：**10 Hz → ~1.6%**；**200 Hz → ~28–33%**（单核、无法自流水）。结论不变：10 Hz 毫无压力，200 Hz 可行但占掉单 BPU 核三分之一。

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

## 真实数据精度（已实测，2026-11-18）

评测集：Parsed 数据集 752 个 `*_exo.csv` trial 抽出的 **1954 个真实窗口**（每 trial 2 个，BT18 的 normal_walk/squat/jump 另加 150 个 dense），全部为完整有限、200 Hz 连续的 249 帧窗口，NaN/断点窗口一律排除、不填充。参考值两条：author 管线 FP32 与 BPU 图 FP32（两者 max diff 1.2e-7，证明预处理等价）。

板上执行用 `hrt_model_exec infer`（见踩坑记录：python 绑定不可用），单位 Nm/kg：

| 模型 | RMSE | p99\|err\| | max\|err\| | \|m\|<0.05 符号一致率 | >0.05 符号一致率 |
|---|---|---|---|---|---|
| `synthcal`（合成校准） | **0.0142** | 0.039 | 0.519（饱和） | 93.8% | **99.9–100%** |
| `realcal`（200 真实窗校准） | 0.0197 | 0.065 | **0.156** | 88.0% | **99.7–100%** |

参考信号 RMS 为 0.31–0.44 Nm/kg（峰值约 ±2），**两个模型的误差都在信号 RMS 的 3–6%**，方向在 |m| > 0.05 Nm/kg 区间几乎完全一致。分活动：normal_walk 0.012–0.013、squat 0.010–0.025、jump 0.019、stairs 0.008–0.016。

值得注意的**反直觉结果**：窄阈值集（synthcal，5.38）在典型幅值上分辨率更高，RMSE 反而更好，但大扭矩窗口会饱和（max 0.519）；宽阈值集（realcal，14.35）典型精度略降、极值可控（max 0.156）。**作为控制输入，推荐 realcal**——误差有界比均值好看更重要。

## 已澄清的两件事


1. **精度** ✅ 已测：见上，真实数据 1954 窗口 RMSE 0.014–0.020 Nm/kg，方向可靠。
2. **板上实测** ✅ 已测：1.4–1.7 ms/次（`hrt_model_exec perf` 与 `infer` 一致，infer 报 1.409 ms）。

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
tools/make_calib.py              合成校准集生成器（固定种子，仅供跑通编译）
tools/make_eval_bundle.py        从 Parsed 抽真实窗口 → 评测包（图输入 + 双 FP32 参考）
tools/make_real_calib.py         真实校准集生成器（每 trial 1 窗，与评测窗错开）
tools/run_eval_cli.py            板上评测（hrt_model_exec 并发，唯一可信路径）
config.makertbin.yaml            makertbin A 配置（正式）；config.b.yaml 为最小回退配置
config.realcal.yaml              真实校准配置（calib_real/，march bayes-e）
.github/workflows/hb_checker.yml    Actions：hb_mapper checker → 日志回推 logs 分支
.github/workflows/hb_makertbin.yml  Actions：编译 → 产物回推 logs-makertbin
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
| 校准集 752 样本把构建拖到 >64 min | `calibration_type: default` 展开成 `['max','kl']` 的 modelwise 搜索，代价对样本数非线性（100 合成样本仅 16.6 s）；`log_level: debug` 还会把整个 `cali_dict` 打印进 docker→tee 管道。→ 改 `calibration_type: max` / 降样本数 / `log_level: info` |
| `calib_ids.txt` 被当样本读：`cannot reshape array of size 16681` | 同上，`cal_data_dir` 全量 glob——provenance 文件必须放在校准目录**之外** |
| **`pyeasy_dnn` 的 `forward()` 返回错误结果** | **`model.forward({name: arr})` 静默忽略数据**（日志 `arg_obj is NOT a container`），位置传参、list、写 `inputs[0].buffer` 也都不对：输出比 FP32 大 10 倍、第二个 batch 槽恒为常数、甚至段错误。同一 `.bin` 用 `hrt_model_exec infer` 跑则与 FP32 `corr=0.9924`。→ **板上评测一律走 CLI**（`tools/run_eval_cli.py`，进程启动 1.25 s，6 并发 ≈160 ms/窗） |
| 图 `[2,12,1,264]` 的 dim0 是 batch | `model_info` 报 `valid shape (2,12,1,264)`，runtime 一次 forward 只更新 batch 0；python 路径下 leg1 输出恒为 27.4292 |

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
| ✅ 拿到可部署 `.bin` | 已完成：`nature_bpu_synthcal.bin` / `nature_bpu_realcal.bin`（各 526 KB） |
| ✅ 板上实测延迟 | 已完成：1.4–1.7 ms/次（`hrt_model_exec perf` 与 `infer` 一致） |
| ✅ 判定 INT8 精度 | 已完成：1954 真实窗口，RMSE 0.014–0.020 Nm/kg，方向一致率 >99.7%（\|m\|>0.05） |
| 决定是否切主线 | 待 owner 裁决。10 Hz CPU 主线本身已达标，BPU 的收益是"回收约 1.6 个 A55 核 + 留余量" |
| 可选：量化余量实验 | `calibration_type: percentile`（如 99.99）或分通道激活，试试同时拿到 synthcal 的 RMSE 和 realcal 的极值 |

## 已知限制

- `nature_bpu_synthcal.bin` 的校准集是合成的，**只可用于延迟测量，精度结论无效，不得流入主仓**。
- 真实数据校准的 `.bin` 命名必须与 synthcal 区分开。
- Actions runner 是临时 VM，产物必须走 artifact 或 `logs` 分支带走。
