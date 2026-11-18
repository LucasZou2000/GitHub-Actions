// sysbench.cpp — RDK X5 系统级实时性压测
//
// 模拟真实部署链路：
//   200 Hz 主循环（IMU 采样节奏）
//     -> 每帧把新采样写进 249 帧环形窗口（shift + 归一化写入）
//     -> 按 infer_every 降频推理（1/2/4 -> 200/100/50 Hz）
//        hbDNNInfer + hbDNNWaitTaskDone（同步，拿到真实完成时刻）
//     -> 80->2 线性 head 得到 4 关节力矩（作为"指令就绪"）
//
// 关键指标：从某帧 IMU 数据到达，到它被消费并产出指令的时间（data_age）。
// 这是端到端延迟里软件的那一段，不含 IMU 总线、CAN 发送和电机机械响应。
//
// 编译：g++ -O2 -std=c++14 -o sysbench sysbench.cpp -I/usr/include/dnn -ldnn -lpthread

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>
#include <sched.h>
#include <fcntl.h>

#include "hb_dnn.h"
#include "hb_sys.h"

static const int LEGS = 2;
static const int CH = 12;
static const int HIST = 249;
static const int PAD = 15;
static const int GLEN = HIST + PAD;   // 264
static const int OUT_CH = 80;
static const int OUT_LEN = 16;

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static long read_long(const char* path) {
  FILE* f = fopen(path, "r");
  if (!f) return -1;
  long v = -1;
  if (fscanf(f, "%ld", &v) != 1) v = -1;
  fclose(f);
  return v;
}

static long rss_kb() {
  FILE* f = fopen("/proc/self/statm", "r");
  if (!f) return -1;
  long size = 0, resident = 0;
  if (fscanf(f, "%ld %ld", &size, &resident) != 2) resident = -1;
  fclose(f);
  return resident * (sysconf(_SC_PAGESIZE) / 1024);
}

struct Stats {
  std::vector<double> v;
  void add(double x) { v.push_back(x); }
  void sort() { std::sort(v.begin(), v.end()); }
  double pct(double p) const {
    if (v.empty()) return 0;
    size_t i = (size_t)(p * (v.size() - 1) + 0.5);
    return v[i];
  }
  double max() const { return v.empty() ? 0 : v.back(); }
  double mean() const {
    if (v.empty()) return 0;
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
  }
};

// ---------- GPIO 打点：给示波器读硬件段延迟用 ----------
// sysfs 每个 pin 打开一次 value 文件，循环里只做 lseek+write（约几 µs）。
struct GpioOut {
  int fd = -1;
  int pin = -1;
  double write_ms = 0.0;   // 单次写入开销，最后要减掉
  long writes = 0;

  bool open_pin(int n) {
    pin = n;
    char path[64];
    snprintf(path, sizeof(path), "/sys/class/gpio/gpio%d", n);
    if (access(path, F_OK) != 0) {
      int e = ::open("/sys/class/gpio/export", O_WRONLY);
      if (e < 0) return false;
      char b[16];
      int len = snprintf(b, sizeof(b), "%d", n);
      if (write(e, b, len) < 0) { /* 已导出也会报错，忽略 */ }
      ::close(e);
    }
    snprintf(path, sizeof(path), "/sys/class/gpio/gpio%d/direction", n);
    int d = ::open(path, O_WRONLY);
    if (d < 0) return false;
    if (write(d, "out", 3) < 0) { ::close(d); return false; }
    ::close(d);
    snprintf(path, sizeof(path), "/sys/class/gpio/gpio%d/value", n);
    fd = ::open(path, O_WRONLY);
    return fd >= 0;
  }

  inline void set(int v) {
    if (fd < 0) return;
    double a = now_s();
    lseek(fd, 0, SEEK_SET);
    if (write(fd, v ? "1" : "0", 1) < 0) { /* ignore */ }
    write_ms += (now_s() - a) * 1e3;
    ++writes;
  }
};

static GpioOut g_tick, g_cmd;   // --gpio-tick / --gpio-cmd

static void usage() {
  fprintf(stderr,
          "usage: sysbench [--hz 200] [--infer-every 1] [--duration 300] [--load 0]\n"
          "                [--model PATH] [--data PATH] [--head PATH] [--tag NAME]\n"
          "                [--rt PRIO] [--pin CPU] [--dump IDX]\n"
          "                [--gpio-tick PIN] [--gpio-cmd PIN]\n");
}

int main(int argc, char** argv) {
  int hz = 200, infer_every = 1, duration = 300, load = 0;
  std::string model = "/home/lucas/bpu_spike/nature_bpu_realcal.bin";
  std::string data_path = "/home/lucas/data-transferrer/chunk_000/graph_inputs.bin";
  std::string head_path = "/home/lucas/data-transferrer/head_f32.bin";
  std::string tag = "run";
  int dump_idx = -1;
  int pin_cpu = -1;
  int rt_prio = 0;   // >0 时切 SCHED_FIFO（需 root 或 CAP_SYS_NICE）
  int gpio_tick = -1, gpio_cmd = -1;   // 示波器打点：数据到达 / 指令就绪

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--hz") hz = atoi(next());
    else if (a == "--infer-every") infer_every = atoi(next());
    else if (a == "--duration") duration = atoi(next());
    else if (a == "--load") load = atoi(next());
    else if (a == "--model") model = next();
    else if (a == "--data") data_path = next();
    else if (a == "--head") head_path = next();
    else if (a == "--tag") tag = next();
    else if (a == "--dump") dump_idx = atoi(next());
    else if (a == "--pin") pin_cpu = atoi(next());
    else if (a == "--rt") rt_prio = atoi(next());
    else if (a == "--gpio-tick") gpio_tick = atoi(next());
    else if (a == "--gpio-cmd") gpio_cmd = atoi(next());
    else { usage(); return 2; }
  }
  if (infer_every < 1) infer_every = 1;

  // ---------- 回放数据 ----------
  FILE* fd = fopen(data_path.c_str(), "rb");
  if (!fd) { fprintf(stderr, "cannot open data %s\n", data_path.c_str()); return 1; }
  fseek(fd, 0, SEEK_END);
  long dbytes = ftell(fd);
  fseek(fd, 0, SEEK_SET);
  size_t win_bytes = (size_t)LEGS * CH * GLEN * sizeof(float);
  size_t wins = (size_t)dbytes / win_bytes;
  if (wins == 0) { fprintf(stderr, "data too small\n"); return 1; }
  std::vector<float> replay((size_t)wins * LEGS * CH * GLEN);
  if (fread(replay.data(), 1, wins * win_bytes, fd) != wins * win_bytes) {
    fprintf(stderr, "short read\n");
    return 1;
  }
  fclose(fd);

  // ---------- head 权重（162 floats: 2x80 weight + 2 bias） ----------
  float headbuf[LEGS * OUT_CH + LEGS];
  FILE* fh = fopen(head_path.c_str(), "rb");
  if (!fh) { fprintf(stderr, "cannot open head %s\n", head_path.c_str()); return 1; }
  if (fread(headbuf, sizeof(float), LEGS * OUT_CH + LEGS, fh) !=
      (size_t)(LEGS * OUT_CH + LEGS)) {
    fprintf(stderr, "bad head file\n");
    return 1;
  }
  fclose(fh);

  // ---------- 模型 ----------
  hbPackedDNNHandle_t packed = nullptr;
  const char* files[1] = {model.c_str()};
  if (hbDNNInitializeFromFiles(&packed, files, 1) != 0) {
    fprintf(stderr, "hbDNNInitializeFromFiles failed\n");
    return 1;
  }
  char const** names = nullptr;
  int32_t name_count = 0;
  hbDNNGetModelNameList(&names, &name_count, packed);
  hbDNNHandle_t dnn;
  if (hbDNNGetModelHandle(&dnn, packed, names[0]) != 0) {
    fprintf(stderr, "hbDNNGetModelHandle failed\n");
    return 1;
  }

  int32_t in_count = 0, out_count = 0;
  hbDNNGetInputCount(&in_count, dnn);
  hbDNNGetOutputCount(&out_count, dnn);
  if (in_count != 1 || out_count != 1) {
    fprintf(stderr, "unexpected tensor count in=%d out=%d\n", in_count, out_count);
    return 1;
  }

  hbDNNTensor input;
  memset(&input, 0, sizeof(input));
  hbDNNGetInputTensorProperties(&input.properties, dnn, 0);
  if (hbSysAllocCachedMem(&input.sysMem[0], input.properties.alignedByteSize) != 0) {
    fprintf(stderr, "alloc input failed\n");
    return 1;
  }
  size_t in_elems = 1;
  for (int d = 0; d < input.properties.validShape.numDimensions; ++d)
    in_elems *= (size_t)input.properties.validShape.dimensionSize[d];

  hbDNNTensor output;
  memset(&output, 0, sizeof(output));
  hbDNNGetOutputTensorProperties(&output.properties, dnn, 0);
  if (hbSysAllocCachedMem(&output.sysMem[0], output.properties.alignedByteSize) != 0) {
    fprintf(stderr, "alloc output failed\n");
    return 1;
  }
  size_t out_elems = 1;
  for (int d = 0; d < output.properties.validShape.numDimensions; ++d)
    out_elems *= (size_t)output.properties.validShape.dimensionSize[d];

  fprintf(stderr, "[sysbench] model=%s in_elems=%zu out_elems=%zu wins=%zu\n",
          names[0], in_elems, out_elems, wins);

  // ---------- 推理控制参数 ----------
  hbDNNInferCtrlParam ctrl;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);
  ctrl.bpuCoreId = 0;   // any core

  // ---------- 环形窗口 ----------
  std::vector<float> ring((size_t)LEGS * CH * GLEN, 0.0f);
  memcpy(ring.data(), replay.data(), win_bytes);   // 起始填满，内容为第一个真实窗口

  // 归一化参数从 preprocess 里预先算好？这里省掉，直接回放已归一化数据。
  // （真实系统的 per-channel 归一化是 24 次乘加，量级 1e-4 ms，不改变结论）

  // ---------- 单帧对拍（验证输出解码与 head 顺序，不参与压测） ----------
  if (dump_idx >= 0 && (size_t)dump_idx < wins) {
    memcpy(input.sysMem[0].virAddr, replay.data() + (size_t)dump_idx * LEGS * CH * GLEN,
           win_bytes);
    hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    hbDNNTaskHandle_t task = nullptr;
    hbDNNTensor* out_ptr = &output;
    hbDNNInfer(&task, &out_ptr, &input, dnn, &ctrl);
    hbDNNWaitTaskDone(task, 200);
    hbDNNReleaseTask(task);
    hbSysFlushMem(&output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    const float* feat = (const float*)output.sysMem[0].virAddr;
    printf("DUMP win=%d\nFEAT", dump_idx);
    for (int k = 0; k < 8; ++k) printf(" %.6f", feat[k * OUT_LEN + (OUT_LEN - 1)]);
    printf("\nLEG1FEAT");
    for (int k = 0; k < 4; ++k)
      printf(" %.6f", feat[((size_t)OUT_CH + k) * OUT_LEN + (OUT_LEN - 1)]);
    printf("\nTORQUE");
    for (int leg = 0; leg < LEGS; ++leg) {
      for (int j = 0; j < LEGS; ++j) {
        double acc = headbuf[LEGS * OUT_CH + j];
        for (int k = 0; k < OUT_CH; ++k)
          acc += (double)feat[((size_t)leg * OUT_CH + k) * OUT_LEN + (OUT_LEN - 1)] *
                 headbuf[j * OUT_CH + k];
        printf(" %.6f", acc);
      }
    }
    printf("\n");
    hbSysFreeMem(&input.sysMem[0]);
    hbSysFreeMem(&output.sysMem[0]);
    hbDNNRelease(packed);
    return 0;
  }

  // ---------- 背景负载 ----------
  auto pin_to = [](int cpu) -> int {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set);
  };
  if (pin_cpu >= 0) {
    if (pin_to(pin_cpu) == 0)
      fprintf(stderr, "[sysbench] main thread pinned to cpu%d\n", pin_cpu);
    else
      fprintf(stderr, "[sysbench] WARN: pin to cpu%d failed\n", pin_cpu);
  }

  std::vector<pid_t> kids;
  for (int i = 0; i < load; ++i) {
    pid_t p = fork();
    if (p == 0) {
      if (pin_cpu >= 0) pin_to(1 + (i % 7));   // 负载避开主线程所在的核
      // 负载进程必须是普通调度，否则会继承父进程的 SCHED_FIFO 抢占主线程
      struct sched_param sp;
      memset(&sp, 0, sizeof(sp));
      sched_setscheduler(0, SCHED_OTHER, &sp);
      double t0 = now_s();
      volatile double x = 0.0;
      while (now_s() - t0 < duration + 5) x += sqrt((double)(rand() % 1000));
      _exit(0);
    }
    if (p > 0) kids.push_back(p);
  }

  // ---------- 主循环 ----------
  const double period = 1.0 / hz;
  const long total_ticks = (long)hz * duration;
  Stats jitter, work, infer_lat, age;
  long late = 0, over45 = 0, over75 = 0, infer_cnt = 0, infer_fail = 0;
  double worst_jitter = 0, worst_age = 0, worst_infer = 0;
  std::vector<double> tick_t0(infer_every, 0.0);

  double next = now_s() + period;
  double t_start = next;

  // 预分配，避免 vector 扩容在中途引入抖动
  jitter.v.reserve((size_t)total_ticks + 8);
  work.v.reserve((size_t)total_ticks + 8);
  infer_lat.v.reserve((size_t)total_ticks / infer_every + 8);
  age.v.reserve((size_t)total_ticks + 8);
  long spike_n = 0;

  // ---------- 实时优先级（放在 fork 之后，避免负载子进程继承） ----------
  if (rt_prio > 0) {
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = rt_prio;
    int rc = sched_setscheduler(0, SCHED_FIFO, &sp);
    if (rc == 0) {
      struct sched_param got;
      int pol = sched_getscheduler(0);
      sched_getparam(0, &got);
      fprintf(stderr, "[sysbench] SCHED_FIFO prio=%d (policy=%d)\n", got.sched_priority, pol);
    } else {
      fprintf(stderr, "[sysbench] WARN: SCHED_FIFO failed (%s), need root/CAP_SYS_NICE\n",
              strerror(errno));
    }
  }

  if (gpio_tick >= 0 || gpio_cmd >= 0) {
    bool ok_t = (gpio_tick < 0) || g_tick.open_pin(gpio_tick);
    bool ok_c = (gpio_cmd < 0) || g_cmd.open_pin(gpio_cmd);
    if (!ok_t || !ok_c)
      fprintf(stderr, "[sysbench] WARN: gpio open failed (tick=%d cmd=%d)\n", gpio_tick, gpio_cmd);
    else
      fprintf(stderr, "[sysbench] gpio markers: tick=%d cmd=%d\n", gpio_tick, gpio_cmd);
    g_tick.set(0);
    g_cmd.set(0);
  }

  float torque[LEGS][2];
  memset(torque, 0, sizeof(torque));

  for (long tick = 0; tick < total_ticks; ++tick) {
    next += period;
    struct timespec ts;
    ts.tv_sec = (time_t)next;
    ts.tv_nsec = (long)((next - (double)ts.tv_sec) * 1e9);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);

    double t0 = now_s();
    g_tick.set(1);   // 示波器：数据到达
    double late_ms = (t0 - next) * 1e3;
    jitter.add(late_ms);
    if (late_ms > 0.5) ++late;
    if (late_ms > 5.0) {
      if (spike_n < 20)
        fprintf(stderr, "[spike] tick=%ld t=%.1fs jitter=%.1fms prev_work=%.1fms\n", tick,
                t0 - t_start, late_ms, work.v.empty() ? 0.0 : work.v.back());
      ++spike_n;
    }

    // 模拟 IMU：把回放窗口的最后一列当作"刚采到的新样本"
    const float* src = replay.data() + (size_t)(tick % wins) * LEGS * CH * GLEN;
    for (int leg = 0; leg < LEGS; ++leg) {
      float* row = ring.data() + (size_t)leg * CH * GLEN;
      for (int c = 0; c < CH; ++c) {
        float* ch_row = row + (size_t)c * GLEN;
        memmove(ch_row, ch_row + 1, (GLEN - 1) * sizeof(float));
        ch_row[GLEN - 1] = src[(size_t)leg * CH * GLEN + (size_t)c * GLEN + (GLEN - 1)];
      }
    }

    // 数据年龄以"计划采样时刻"为起点：真实系统里 IMU 按固定节拍采样，
    // 主循环被推迟时数据就是积压了，必须计入延迟。
    tick_t0[tick % infer_every] = next;

    if (tick % infer_every == 0) {
      memcpy(input.sysMem[0].virAddr, ring.data(), win_bytes);
      hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

      double ti0 = now_s();
      hbDNNTaskHandle_t task = nullptr;
      hbDNNTensor* out_ptr = &output;
      int32_t ret = hbDNNInfer(&task, &out_ptr, &input, dnn, &ctrl);
      if (ret != 0 || task == nullptr) {
        ++infer_fail;
      } else {
        ret = hbDNNWaitTaskDone(task, 200);
        hbDNNReleaseTask(task);
        if (ret != 0) ++infer_fail;
      }
      hbSysFlushMem(&output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

      const float* feat = (const float*)output.sysMem[0].virAddr;
      for (int leg = 0; leg < LEGS; ++leg) {
        for (int j = 0; j < LEGS; ++j) {
          double acc = headbuf[LEGS * OUT_CH + j];
          for (int k = 0; k < OUT_CH; ++k) {
            // 输出 [2,80,1,16]，取 [leg, k, 0, 15]
            double f = feat[((size_t)leg * OUT_CH + k) * OUT_LEN + (OUT_LEN - 1)];
            acc += f * headbuf[j * OUT_CH + k];
          }
          torque[leg][j] = (float)acc;
        }
      }

      double ti1 = now_s();
      g_cmd.set(1);    // 示波器：力矩就绪
      double lat = (ti1 - ti0) * 1e3;
      infer_lat.add(lat);
      if (lat > worst_infer) worst_infer = lat;
      ++infer_cnt;

      // 本推理周期内每一帧数据到"指令就绪"的年龄
      for (int b = 0; b < infer_every; ++b) {
        double t_frame = tick_t0[b];
        if (t_frame <= 0) continue;
        double a = (ti1 - t_frame) * 1e3;
        age.add(a);
        if (a > worst_age) worst_age = a;
        if (a > 45.0) ++over45;
        if (a > 75.0) ++over75;
      }
    }

    g_cmd.set(0);
    g_tick.set(0);

    double w = (now_s() - t0) * 1e3;
    work.add(w);

    if (tick > 0 && tick % (hz * 30) == 0) {
      fprintf(stderr,
              "[%4lds] tick=%ld late=%ld infer=%ld fail=%ld rss=%ldKB ddr=%.1fC cpu=%.1fC freq=%ldMHz\n",
              tick / hz, tick, late, infer_cnt, infer_fail, rss_kb(),
              read_long("/sys/class/thermal/thermal_zone0/temp") / 1000.0,
              read_long("/sys/class/thermal/thermal_zone1/temp") / 1000.0,
              read_long("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") / 1000);
    }
  }

  double elapsed = now_s() - t_start;
  jitter.sort(); work.sort(); infer_lat.sort(); age.sort();

  printf("\n=== %s | %d Hz loop, infer every %d (= %d Hz), %ds, load=%d ===\n",
         tag.c_str(), hz, infer_every, hz / infer_every, duration, load);
  printf("actual rate      : %.3f Hz (%ld ticks in %.3f s)\n", (double)total_ticks / elapsed,
         total_ticks, elapsed);
  printf("late ticks       : %ld (%.4f%%) [>0.5ms behind schedule]\n", late,
         100.0 * late / total_ticks);
  printf("tick jitter ms   : p50=%.3f p99=%.3f max=%.3f\n", jitter.pct(0.50),
         jitter.pct(0.99), jitter.max());
  printf("frame work ms    : p50=%.3f p99=%.3f max=%.3f\n", work.pct(0.50),
         work.pct(0.99), work.max());
  printf("infer latency ms : n=%ld p50=%.3f p95=%.3f p99=%.3f max=%.3f fail=%ld\n",
         infer_cnt, infer_lat.pct(0.50), infer_lat.pct(0.95), infer_lat.pct(0.99),
         worst_infer, infer_fail);
  printf("data->cmd ms     : n=%zu p50=%.3f p95=%.3f p99=%.3f max=%.3f\n",
         age.v.size(), age.pct(0.50), age.pct(0.95), age.pct(0.99), worst_age);
  printf("budget           : >45ms %ld (%.3f%%)  >75ms %ld (%.4f%%)\n", over45,
         100.0 * over45 / (double)(age.v.size() ? age.v.size() : 1), over75,
         100.0 * over75 / (double)(age.v.size() ? age.v.size() : 1));
  printf("sched spikes     : %ld ticks >5ms late\n", spike_n);
  printf("peak duty        : infer p99 %.1f%% of period, work p99 %.1f%%\n",
         100.0 * infer_lat.pct(0.99) / (period * 1e3),
         100.0 * work.pct(0.99) / (period * 1e3));
  printf("rss end          : %ld KB\n", rss_kb());
  if (g_tick.writes + g_cmd.writes > 0) {
    long n = g_tick.writes + g_cmd.writes;
    double tot = g_tick.write_ms + g_cmd.write_ms;
    printf("gpio overhead    : %.4f ms/write avg over %ld writes "
           "(subtract %.3f ms/tick from data->cmd when reading a scope)\n",
           tot / (double)n, n, tot / (double)(total_ticks ? total_ticks : 1));
  }

  for (pid_t p : kids) waitpid(p, nullptr, 0);
  hbSysFreeMem(&input.sysMem[0]);
  hbSysFreeMem(&output.sysMem[0]);
  hbDNNRelease(packed);
  return 0;
}
