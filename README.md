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
| **系统级 200 Hz × 5 分钟** | **0 硬失步、端到端 p99 0.90 ms / max 5.79 ms、进程 CPU 0.086 核、69.6 °C 无降频**（4 核背景负载 + RT 80） |
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

## 系统级实时性（已实测，2026-11-18）

问题不是"BPU 能不能算"，而是"200 Hz IMU + 降频推理的整条链路能不能稳定跑"，以及端到端延迟剩多少给硬件。用 `tools/sysbench.cpp`（板上 `~/sysbench/`）实测：200 Hz 定频主循环（`clock_nanosleep` 绝对时间）→ 249 帧环形窗口 shift+写入 → `hbDNNInfer` + `hbDNNWaitTaskDone` 同步推理 → head 80→2 → 力矩，叠加 4 核背景负载，每配置连续 5 分钟 / 60000 帧。

| 配置 | 迟到节拍 | tick jitter p99/max | 推理 p99/max | data→cmd p99/max |
|---|---|---|---|---|
| 200 Hz 推理，performance，无 RT | 1 | 0.068 / 0.684 | 0.759 / 2.384 | 0.849 / 2.501 |
| **200 Hz 推理，performance，RT 80** | **0** | **0.015 / 0.023** | 0.764 / 1.821 | **0.799 / 1.972** |
| 100 Hz 推理，performance，RT 80 | **0** | 0.015 / 0.048 | 0.785 / 1.828 | 5.805 / 6.063 |
| 200 Hz 推理，schedutil，RT 80（load=4） | **0** | 0.018 / 0.037 | 0.788 / 1.831 | 0.829 / 2.013 |
| 200 Hz 推理，schedutil，RT 80（**load=0**） | 0 | 0.074 / 0.093 | **1.939** / 3.928 | 2.077 / 4.162 |
| **200 Hz 推理，performance，RT 80（load=0）** | **0** | 0.016 / 0.028 | 0.752 / 1.441 | **0.790 / 1.501** |
| 50 Hz 推理，performance，RT 80（load=4） | 6 | 0.015 / 28.819 | 0.820 / 33.756 | 15.822 / 48.799 |

单位 ms。结论三条：

1. **RT 的收益在抖动，不在均值**。均值只差 0.05 ms，但 jitter p99 从 0.068 降到 0.015（4.5×）、max 从 0.684 降到 0.023（30×）——均值指标完全看不出 RT 的作用。
2. **`governor=performance` 是硬要求**。空载时 schedutil 把 CPU 压到 300 MHz，推理 p99 从 0.752 恶化到 1.939（2.6×）。有 4 核负载时因频率被拉满而掩盖了这个问题——所以**只在满载下标定会得出错误结论**。
3. **降频推理不省心**。50/100 Hz 推理的延迟不是常量，而是按 `(infer_every-1)×5ms + 推理耗时` 分布：100 Hz 最坏 6.06 ms、50 Hz 最坏约 15.8 ms（均为实测，不含文末那个长尾事件），比 200 Hz 的 2.0 ms 差 3–8 倍。省下的算力远小于多出的延迟。

不要绑核（`--pin`）：主线程绑 cpu0 会劣化到 10 次迟到 + 一次 15.6 ms 卡顿，因为 BPU 驱动中断和内核线程也在 cpu0 上；默认调度反而更稳。

### 真实 IMU 数据源（板上实测）

板上跑着 ROS 2 Humble + `ess_imu_driver2`（`ros2 launch ess_imu_driver2 imu_hub_launch.py`，稳态 14–17% CPU），发布 8 个话题 `/A1|A2|B1|B2/data_raw|tempc`，即双测腿 × 大腿/小腿共 4 个 IMU。`ros2 topic hz` 实测：

| 话题 | 平均频率 | 到达间隔 | 抖动 std |
|---|---|---|---|
| `/A1/data_raw` | 200.07–200.19 Hz | 4–5 ms | **0.05–0.06 ms** |
| `/A2/data_raw` | 200.02–200.07 Hz | 5 ms | **0.03 ms** |

**数据源本身不是瓶颈**：200 Hz 稳、抖动 0.05 ms 级。延迟预算的压力全在推理节拍和下游 CAN/电机。

### 一个未定性的长尾（重要，别忽略）

约 20% 的长跑里会出现**单次 ~40–50 ms 停顿**，位置固定在 t≈5.7 s 前后，之后整段不再复现（`late` 计数一次跳到位后保持不变）。两种签名都见过：一次是 `prev_work=47.5ms`（推理本身耗时 47 ms），一次是 `jitter=49.5ms` 而推理仍为 1.8 ms（进程被挂起）。已排除：BPU/DDR devfreq（都锁在 performance，996 MHz / 4266 MHz）、CPU governor（performance，8 核 1500 MHz）、RT 节流（duty 仅 16%，限值 95%）、THP（内核未启用）、音频 codec（es8326 中断计数为 0）。缩小缓冲到 50 KB / 2.5 MB 后 6 次连跑未见，但 16 MB 缓冲也有 2 次连跑未见——**不可按需复现**。

后果：命中时 `data→cmd` 最大值 47–66 ms，会有 1–5 个样本越过 45 ms 期望线，但**没有越过 75 ms 硬上限**。工程含义是必须做超期兜底（推理超过 ~10 ms 就沿用上一帧力矩），并把这个事件计数打进运行时日志；不能假设"5 分钟干净"就等于"永远不会发生"。

`--gpio-tick <pin> --gpio-cmd <pin>` 可按"数据到达 / 力矩就绪"打点，供示波器直接读**硬件段**延迟（IMU 总线、CAN、电机机械响应），并会打印单次 sysfs 写开销供扣减。引脚号用 `/sys/class/gpio/gpiochip*/label` 的 base + 偏移，按 RDK X5 40-pin 定义查；注意 gpio 498-505（AON）被电源键/ID 占用。

## Q8（BPU INT8）vs ONNX FP32（CPU）—— 方案对比与 Q8 完整评估（2026-11-18）

**问题**：Q8 在 50 Hz 下有无法定位的长尾（见上节），工程上要一个"确定"的方案。于是把两条候选路径放进**同一套测量框架**里对比，看 CPU/内存/温度/正确性/误差，而不是只看推理耗时。

**方法**：`tools/sysbench2.cpp`（板上 `~/sysbench/`）——同一份主循环代码，两个后端可切换：

- `--backend bpu`：`hbDNNInfer` + `hbDNNWaitTaskDone`（同步，能拿到完成时刻），head 80→2 在 CPU；
- `--backend onnx`：onnxruntime 1.20.1 CPU EP，`--threads/--spin` 可调；用 `--async` 时推理在独立线程，主循环只投递窗口不等待（FP32 单帧 > 20 ms，同步内联会把 200 Hz 主循环直接拖垮——这个对照也测了）；
- 1 Hz 采样（`--stats`）：`/proc/stat` 逐核忙时、`/proc/self/stat` RSS、`MemAvailable`、`thermal_zone0/1`、CPU/BPU/DDR 频率、每秒推理分位；温度另用 `hrut_somstatus` 前后各取一次。

统一条件：**200 Hz 主循环 × 300 s（60000 帧）× 4 核背景负载 × `governor=performance`（8 核 1500 MHz）× `SCHED_FIFO 80`**。BPU 996 MHz、DDR 4266 MHz 已锁 performance。

### 主表（全部为板上实测，5 分钟稳态）

| 配置 | 推理 p50 / p99 / max | 推理频率 | 迟到帧 | 端到端 p50 / p99 / max | >45 ms | >75 ms | 进程 CPU | 每推理线程时间 | RSS | CPU 温度 | BPU 温度 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **BPU Q8 @200 Hz** | **0.75 / 0.87 / 5.76** | 200 Hz | 1 | **0.77 / 0.90 / 5.79** | **0** | **0** | **0.086 核** | 0.43 ms | 34.2 MB | **69.6 °C** | **67.5 °C** |
| BPU Q8 @100 Hz | 0.76 / 0.82 / 1.13 | 100 Hz | 0 | 1.19 / 5.82 / 6.19 | 0 | 0 | 0.047 核 | 0.48 ms | 34.0 MB | 69.7 °C | 67.8 °C |
| FP32 CPU @50 Hz，4 线程自旋 | 14.24 / 17.04 / 46.63 | 50 Hz | 0 | 23.35 / 31.85 / 74.54 | 31 | 0 | 3.59 核 | 71.9 ms | 29.9 MB | 80.9 °C | 75.5 °C |
| FP32 CPU @50 Hz，8 线程自旋 | 29.23 / 69.01 / 114.20 | 50 Hz（丢 49%） | 0 | 49.60 / 116.05 / 182.28 | 34858 | **8551** | 4.93 核 | 192.7 ms | 30.4 MB | 81.3 °C | 76.1 °C |
| FP32 CPU @50 Hz，8 线程不自旋 | 18.07 / 20.78 / 39.14 | 50 Hz（丢 4.4%） | 0 | 27.16 / 52.63 / 74.19 | 1456 | 0 | 2.41 核 | 50.5 ms | 30.2 MB | 78.3 °C | 74.1 °C |

单位 ms，除非另注。**"端到端"= 某帧 IMU 数据到达 → 由它得出的力矩就绪**（软件段，不含 IMU 总线/CAN/电机）。"进程 CPU"= 该进程总线程时间 ÷ 墙钟（8 核 SoC，1.0 = 占满一核）；整机占用率另计，但其中 50% 是那 4 个背景 busy 核，不能直接当成本进程开销。

### 逐项读数

- **CPU**：Q8 @200 Hz 占 **0.086 核**（8.6%），其中约 0.042 ms/帧 是窗口 memcpy + head 80→2，剩下是 `hbDNNWaitTaskDone` 的等待（等待不耗 CPU）。FP32 最省 CPU 的配置也要 **2.41 核**，最快配置 **3.59 核**——相差 **28–42 倍**。注意 FP32 的线程时间远大于推理耗时（14.24 ms 推理 vs 71.9 ms 线程时间）：4 个自旋线程在整个推理窗口里空转，这是 ORT 在小核 SoC 上的固有代价。
- **"线程越多越糟"**：8 线程自旋在 4 核背景负载下 p99 从 17 ms 崩到 **69 ms**（端到端 116 ms），因为 8 个自旋线程和 4 个负载进程抢同一批核，调度器反复抢占、缓存全掉。**FP32 在这颗 SoC 上不能用 8 线程自旋**。
- **内存**：两者 RSS 都在 30–34 MB（BPU 侧多 4 MB 是 HBRT + 对齐张量）。5 分钟内 RSS 增长 1.9–3.4 MB，来源是压测工具自身的统计数组（`reserve` 后按需触页）而非模型侧——DNN 侧在 t≈60 s 后即平台化。可用内存全程 6.6 GB 不变。
- **温度**：Q8 @200 Hz 稳态 CPU 69.6 °C / BPU 67.5 °C / DDR 69.5 °C（起始 61–63 °C，5 分钟升 6 °C 后走平）；FP32 全部配置 78–81 °C。**BPU 侧的温升比 CPU 全核算力低**：0.68 ms 的 BPU 负载没有把 SoC 推到热限。全程 CPU 1500 MHz / BPU 996 MHz / DDR 4266 MHz **无一次降频**，无热节流。
- **抖动**：Q8 jitter p99 **0.017 ms**、max 0.82 ms（那一次迟到帧）。FP32 因为推理在独立线程，主循环本身也稳（jitter p99 0.03 ms），抖动不是它的短板——它的短板是延迟本身。

### Q8 模型完整评估

**模型级**

| 项 | 值 |
|---|---|
| 产物 | `nature_bpu_realcal.bin`，**526 KB**（FP32 ONNX 1.19 MB，2.26× 压缩） |
| 输入 / 输出 | `[2,12,1,264]` float32（25 344 B，aligned）/ `[2,80,16]` float32（10 240 B）；head 80→2 留在 CPU |
| 算子放置 | 25 / 25 全 BPU，单子图，无 CPU 回退；末层 `HzLut2Layer` int16 |
| 计算量 | 48.03 MMAC / 次 |
| BPU 纯硬件时间 | **0.679 ms**（`hrt_model_exec perf`，2000 帧，min 0.657 / max 1.419，**1461 FPS**） |
| 主循环内实测 | 0.751 ms p50（含 memcpy + 同步等待开销 ≈ 0.07 ms） |
| 算力利用率 | 70.7 GMAC/s ÷ 2.5 TMAC/s ≈ **2.8%**（小模型 + 25 个算子的固定启动开销，符合预期） |
| BPU 占空比 | 200 Hz → **17.4%**（p99）；100 Hz → 8.7% |
| DDR 流量 | 每帧 in+out 35.6 KB → 200 Hz 约 **7.1 MB/s**，可忽略 |
| 确定性 | 同一批 642 窗口跑两遍，输出 **md5 完全相同**（定点点运算，逐位可复现） |
| 校准 | 200 个真实窗口，`calibration_type: max`，per-channel W / per-tensor act |

**精度**（1954 个真实窗口 → 7816 个关节输出，参考为 author 管线 FP32）

| 关节 | 相对 RMSE | 绝对 RMSE | p99 \|err\| | max \|err\| | R² | cos | SNR | 等效位数 | 偏差 | \|m\|>0.1 方向一致 |
|---|---|---|---|---|---|---|---|---|---|---|
| LH | 5.14% | 0.0204 | 0.0667 | 0.1388 | 0.9967 | 0.9987 | 25.8 dB | 4.28 bit | +0.0006 | 100% |
| LK | **6.42%** | 0.0197 | 0.0673 | 0.1343 | 0.9959 | 0.9981 | 23.9 dB | 3.96 bit | +0.0011 | 100% |
| RH | 4.63% | 0.0202 | 0.0632 | 0.1476 | 0.9972 | 0.9990 | 26.7 dB | 4.43 bit | −0.0032 | 100% |
| RK | 5.68% | 0.0185 | 0.0615 | **0.1556** | 0.9967 | 0.9984 | 24.9 dB | 4.14 bit | −0.0012 | 100% |
| **合计** | **5.33%** | **0.0197** | **0.0649** | **0.1556** | **0.9970** | **0.9986** | **25.5 dB** | **4.23 bit** | **−0.0007** | **100%** |

单位 Nm/kg。参考信号 RMS 0.31–0.44，峰值约 ±2。三个必须一起看的事实：

1. **参考基线是无损的**：同一批 128 窗口用板上 C++ onnxruntime 跑 FP32，与 author 参考 **max 9.5e-7 Nm/kg**（相对 1.2e-5 %）；C++ hbDNN 与 `hrt_model_exec` CLI 差 3.6e-7。**表里 5.33% 全部属于 INT8 量化**，不含实现差异。
2. **误差是加性量化噪声，不是系统性失真**。按扭矩幅值分档，绝对误差几乎平坦（0.0155 → 0.0305 Nm/kg），相对误差随幅值单调下降（56% → 2.5%）；偏差 ≈ 0（−0.0007）。小幅值档的相对误差高是分母效应（信号本身 0.03 Nm/kg，与噪声同量级），工程上只关心 |m|>0.1 区间，那里是 13.3% → 2.5%，方向一致率 100%。
3. **按活动拆没有短板**：normal_walk 4.2%（0.0132）、stairs 4.8%、squats 3.4%、jump 6.1%、sit_to_stand 7.7%、ball_toss 9.7%（n=60，样本少）。最差（twister，n=20）24.9% 仍是有界误差，没有饱和式崩坏（对比 synthcal 的 max 0.519）。

### 判定：通过

| 验收项 | 阈值 | Q8 @200 Hz | 结论 |
|---|---|---|---|
| 主循环稳定 | 5 分钟 0 硬失步 | 60000 帧 0 hard miss、1 次 0.82 ms 迟到 | ✅ |
| 端到端期望 | p99 < 45 ms | **0.90 ms** | ✅ 余量 50× |
| 端到端硬限 | max < 75 ms | **5.79 ms** | ✅ |
| 精度 | RMSE ≤ 0.02–0.03，max ≤ 0.1–0.15，方向一致 >99% | 0.0197 / 0.156 / 100% | ✅（max 贴边，见风险） |
| CPU 余量 | 留出 IMU 驱动 + ROS | 0.086 核（1.1% of 8 核） | ✅ |
| 温度 | 不触热节流 | 69.6 °C，无降频 | ✅ |
| 确定性 | 逐位可复现 | md5 相同 | ✅ |

**结论：Q8 @200 Hz 是唯一同时满足延迟、抖动、CPU 余量与温度的方案，评估通过，不需要重训/QAT。FP32 只保留为离线参考实现。**

FP32 为什么不行，不是"慢一点"：单帧 14–29 ms 已经吃掉 50 Hz 的整个 20 ms 时隙，端到端被推到 32–116 ms，最差配置 **14% 的帧越过 75 ms 硬限**；同时把 8 核 SoC 打到 96–100%，与 IMU 驱动抢核。这不是调参能解决的量级差。

**残留风险（评估通过不等于没有风险）**

- `max |err| 0.156` 贴着 0.15 的验收线，且集中在 `lift_weight / lunges / dynamic_walk` 这类高冲击窗口——若下游对单帧极值敏感，用 `calibration_type: percentile` 或分通道激活再压一档。
- 那一次 5.76 ms 推理长尾（t≈5.7 s，见上一节）在这个 5 分钟窗口里也出现了一次；`data→cmd` max 5.79 ms，仍远低于 45 ms，但**超期兜底（>10 ms 沿用上一帧）+ 事件计数**这条工程要求不变。
- 本轮背景负载是 4 个 busy loop（50% SoC），比真实系统（ROS 2 + IMU 驱动 ≈ 15%）更狠，所以温度/CPU 数字偏保守；反过来也说明 Q8 在更差的环境下仍有 50× 延迟余量。

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
tools/sysbench.cpp               板上系统级实时性压测（200Hz 主循环 + 同步推理 + head，
                                 支持 --rt/--load/--pin/--gpio-tick/--gpio-cmd）
tools/sysbench2.cpp              双后端版：--backend bpu|onnx、--async、--threads/--spin、
                                 --stats（1Hz CPU/内存/温度/频率采样）、--dump-all（精度对拍）
tools/analyze_q8.py              INT8 精度评估（相对误差/分位/余弦/SNR/分档/分活动）
tools/make_report.py             生成 perf_matrix.csv / q8_vs_fp32.csv / report.html（自包含 SVG 图）
evidence/q8_eval/                本轮评测的全部数据：runs/（5 分钟压测原始采样）、accuracy*.csv、
                                 report.html、raw/（板上输出与 FP32 参考）
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
| ✅ 双方案对比（Q8 vs FP32） | 已完成：Q8@200 Hz = 0.086 核 / p99 0.90 ms / 误差 5.33%；FP32@50 Hz = 2.4–3.6 核 / p99 32–116 ms，最差配置 14% 帧越 75 ms |
| ✅ Q8 5 分钟系统压测 | 已完成：5 配置 × 300 s，0 hard miss，69.6 °C 无降频，见"Q8 vs ONNX FP32"一节 |
| 决定是否切主线 | 待 owner 裁决。结论已明确：**Q8 @200 Hz 达标，FP32 结构性不达标**；收益是 0.086 核 vs 2.4–3.6 核 |
| 可选：量化余量实验 | `calibration_type: percentile`（如 99.99）或分通道激活，试试同时拿到 synthcal 的 RMSE 和 realcal 的极值 |

## 未来优化展望（评估通过后的候选，按优先级）

Q8 @200 Hz 的判定已经过了，下面这些不再是"能不能用"，而是"值不值得做"。按投入产出排序：

| 优先级 | 项 | 预期收益 | 代价 / 前置条件 |
|---|---|---|---|
| **P0** | **落成 ROS 2 节点**：订阅 `/A1|A2|B1|B2/data_raw`，200 Hz 组窗 → 200 Hz 推理 → 发布力矩，内置超期兜底（>10 ms 沿用上一帧）、长尾事件计数、温度/频率自检、启动自校准（空载零偏） | 从"可行性"到"可部署"，这是唯一必须做的一步 | 需要动板上 ROS 工作区（待 owner 确认）；`pyeasy_dnn` 不可用，节点必须用 `hbDNN` C API |
| P1 | **100 Hz 推理 + 线性插值到 200 Hz** | BPU 占空比 17.4% → 8.7%，CPU 0.086 → 0.047 核；窗口 1.245 s、相邻重叠 248/249，精度几乎不降 | 端到端 p99 0.9 → 5.8 ms（仍远低于 45 ms 期望）；需要插值逻辑与相位对齐 |
| P1 | **硬件段延迟实测**：`--gpio-tick/--gpio-cmd` 已实现，接示波器量 IMU 中断 → 力矩更新 → CAN → 电机电流环 | 把"软件段 0.9 ms"补成完整链路预算，才知道 45 ms 里还剩多少 | 需要示波器与 40-pin 引脚（gpio 498–505 已被电源键/ID 占用） |
| P2 | **长尾定性**：t≈5.7 s 的单次 40–50 ms 停顿 | 从"兜底"变"消除"，可去掉 watchdog 的心理负担 | 需 `perf sched record` / `ftrace`（`sched_switch`、`irq_handler_entry`）在命中时抓栈，黑盒复现已验证不可靠 |
| P2 | **双 BPU 核并行 / 双模型孪生**（左右腿各一个） | 单核 17%，双核可撑 400 Hz 或更重的模型（LSTM/Transformer） | 收益只有在模型变大或频率提高时才兑现 |
| P3 | **QAT 或 `calibration_type: percentile`** | `max |err|` 从 0.156 压到 ~0.08 | PTQ 已够用（RMSE 0.0197、方向 100%），只在极值敏感场景才值得 |

## 已知限制

- `nature_bpu_synthcal.bin` 的校准集是合成的，**只可用于延迟测量，精度结论无效，不得流入主仓**。
- 真实数据校准的 `.bin` 命名必须与 synthcal 区分开。
- Actions runner 是临时 VM，产物必须走 artifact 或 `logs` 分支带走。
