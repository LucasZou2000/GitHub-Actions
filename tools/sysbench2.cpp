// sysbench2.cpp — RDK X5 双后端系统级压测（BPU INT8 vs ONNX FP32 CPU）
//
// 与 sysbench.cpp 同一套链路模型：
//   200 Hz 主循环（IMU 采样节奏）
//     -> 每帧把新采样写进环形窗口（shift + 写入）
//     -> 按 infer_every 降频推理（1/2/4 -> 200/100/50 Hz）
//     -> 力矩就绪
//
// 新增：
//   --backend bpu|onnx     推理后端（bpu: hbDNN 同步 API；onnx: onnxruntime CPU）
//   --threads N --spin 0|1 onnxruntime intra-op 线程数与自旋
//   --stats CSV            1 Hz 资源采样（CPU/内存/温度/频率/窗口内推理分位）
//   --dump-all PATH        逐窗口跑一遍输出 [4] float32，用于精度对拍
//
// 编译：
//   g++ -O2 -std=c++17 -o sysbench2 sysbench2.cpp -I/usr/include/dnn -I./orthead -ldnn -lpthread \
//     <ort>/libonnxruntime.so.1.20.1 -Wl,-rpath,<ort>

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
#include <sys/resource.h>
#include <time.h>
#include <sched.h>
#include <fcntl.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

#include "hb_dnn.h"
#include "hb_sys.h"
#include "onnxruntime_cxx_api.h"

static const int LEGS = 2;
static const int CH = 12;
static const int PAD = 15;
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

static long mem_avail_kb() {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return -1;
  char line[256];
  long v = -1;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "MemAvailable:", 13) == 0) { v = atol(line + 13); break; }
  }
  fclose(f);
  return v;
}

static double proc_cpu_s() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return (double)ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 +
         (double)ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

// /proc/stat：聚合 + 逐核忙时（jiffies）
struct CpuTimes {
  unsigned long long agg_busy = 0, agg_tot = 0;
  unsigned long long core_busy[16], core_tot[16];
  int n = 0;
  bool read() {
    FILE* f = fopen("/proc/stat", "r");
    if (!f) return false;
    char line[512];
    bool first = true;
    n = 0;
    while (fgets(line, sizeof(line), f)) {
      if (strncmp(line, "cpu", 3) != 0) break;
      unsigned long long v[8] = {0};
      if (first) {
        sscanf(line + 3, "%llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2],
               &v[3], &v[4], &v[5], &v[6], &v[7]);
        agg_busy = v[0] + v[1] + v[2] + v[5] + v[6] + v[7];
        agg_tot = agg_busy + v[3] + v[4];
        first = false;
      } else {
        sscanf(line + 3, "%*d %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2],
               &v[3], &v[4], &v[5], &v[6], &v[7]);
        if (n < 16) {
          core_busy[n] = v[0] + v[1] + v[2] + v[5] + v[6] + v[7];
          core_tot[n] = core_busy[n] + v[3] + v[4];
          ++n;
        }
      }
    }
    fclose(f);
    return true;
  }
};

struct Stats {
  std::vector<double> v;
  void reserve(size_t n) { v.reserve(n); }
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

// ---------- 1 Hz 资源采样 ----------
struct ResMon {
  FILE* csv = nullptr;
  CpuTimes prev;
  double t_origin = 0;
  double prev_t = 0, prev_cpu_s = 0;
  long long prev_infer = 0;
  long rss_start = -1, rss_peak = 0;
  double mem_min_mb = 1e9;
  double freq_min = 1e9, freq_max = 0;
  Stats proc_pct, sys_pct, core_max_pct, t_cpu, t_ddr, freq0, alloc_mb;

  // 秒窗内的推理分位
  std::vector<double> win;

  void begin(const char* path) {
    prev.read();
    t_origin = now_s();
    prev_t = t_origin;
    prev_cpu_s = proc_cpu_s();
    rss_start = rss_kb();
    if (path && *path) {
      csv = fopen(path, "w");
      if (csv)
        fprintf(csv,
                "t_s,cpu_proc_pct,cpu_sys_pct,cpu_core_max_pct,cpu0_mhz,bpu_mhz,ddr_mhz,"
                "rss_kb,memavail_mb,t_cpu_c,t_ddr_c,infer_p50_ms,infer_max_ms,infer_n\n");
    }
  }

  void tick(double t_abs, long long infer_n, bool force_print) {
    double dt = t_abs - prev_t;
    if (dt < 1.0) return;
    double t = t_abs - t_origin;
    CpuTimes cur;
    cur.read();
    double dproc = proc_cpu_s() - prev_cpu_s;
    double proc_pct_v = 100.0 * dproc / dt;

    double dbusy = (double)(cur.agg_busy - prev.agg_busy);
    double dtot = (double)(cur.agg_tot - prev.agg_tot);
    double sys_pct_v = dtot > 0 ? 100.0 * dbusy / dtot : 0;

    double core_max = 0;
    for (int i = 0; i < cur.n && i < prev.n; ++i) {
      double cb = (double)(cur.core_busy[i] - prev.core_busy[i]);
      double ct = (double)(cur.core_tot[i] - prev.core_tot[i]);
      if (ct > 0) core_max = std::max(core_max, 100.0 * cb / ct);
    }

    long f0 = read_long("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    long bpu = read_long("/sys/class/devfreq/3a000000.bpu/cur_freq");
    long ddr = read_long("/sys/class/devfreq/soc:ddrc-freq/cur_freq");
    long tc = read_long("/sys/class/thermal/thermal_zone1/temp");   // thermal-cpu
    long td = read_long("/sys/class/thermal/thermal_zone0/temp");   // thermal-ddr
    long rss = rss_kb();
    double mav = mem_avail_kb() / 1024.0;

    double f0mhz = f0 > 0 ? f0 / 1000.0 : 0;
    double bpumhz = bpu > 0 ? bpu / 1e6 : 0;
    double ddrmhz = ddr > 0 ? ddr / 1e6 : 0;
    double tcv = tc > 0 ? tc / 1000.0 : 0;
    double tdv = td > 0 ? td / 1000.0 : 0;

    proc_pct.add(proc_pct_v);
    sys_pct.add(sys_pct_v);
    core_max_pct.add(core_max);
    if (f0mhz > 0) { freq0.add(f0mhz); freq_min = std::min(freq_min, f0mhz); freq_max = std::max(freq_max, f0mhz); }
    if (tcv > 0) t_cpu.add(tcv);
    if (tdv > 0) t_ddr.add(tdv);
    rss_peak = std::max(rss_peak, rss);
    mem_min_mb = std::min(mem_min_mb, mav);

    double p50 = 0, mx = 0;
    if (!win.empty()) {
      std::vector<double> w = win;
      std::sort(w.begin(), w.end());
      p50 = w[w.size() / 2];
      mx = w.back();
    }
    if (csv)
      fprintf(csv, "%.0f,%.1f,%.1f,%.1f,%.0f,%.0f,%.0f,%ld,%.0f,%.1f,%.1f,%.3f,%.3f,%lld\n",
              t, proc_pct_v, sys_pct_v, core_max, f0mhz, bpumhz, ddrmhz, rss, mav, tcv, tdv,
              p50, mx, infer_n);
    if (csv) fflush(csv);

    if (force_print)
      fprintf(stderr, "[%4.0fs] proc=%.0f%% sys=%.0f%% coremax=%.0f%% cpu=%.0fC ddr=%.0fC "
                      "f=%.0fMHz rss=%ldKB infer_p50=%.2fms n=%lld\n",
              t, proc_pct_v, sys_pct_v, core_max, tcv, tdv, f0mhz, rss, p50, infer_n);

    win.clear();
    prev = cur;
    prev_t = t_abs;
    prev_cpu_s = proc_cpu_s();
    (void)prev_infer;
  }

  void finish() { if (csv) { fclose(csv); csv = nullptr; } }
};

// ---------- 异步推理线程（真实部署形态：推理不占主循环） ----------
struct AsyncInfer {
  std::thread th;
  std::mutex mu;
  std::condition_variable cv;
  bool stop = false, has_job = false, busy = false;
  std::vector<float> stage;
  long long job_tick = -1;
  std::atomic<long long> done_count{0}, drop_count{0};
  float result[4] = {0, 0, 0, 0};
  double t_done = 0, lat_ms = 0;
  std::function<int(const float*, float*)> fn;

  void start(std::function<int(const float*, float*)> f, size_t n) {
    fn = std::move(f);
    stage.assign(n, 0.0f);
    th = std::thread([this] { loop(); });
  }

  void loop() {
    for (;;) {
      long long jt = -1;
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this] { return stop || has_job; });
        if (stop) return;
        has_job = false;
        busy = true;
        jt = job_tick;
      }
      float o[4] = {0, 0, 0, 0};
      double a = now_s();
      fn(stage.data(), o);
      double b = now_s();
      {
        std::lock_guard<std::mutex> lk(mu);
        memcpy(result, o, sizeof(o));
        t_done = b;
        lat_ms = (b - a) * 1e3;
        last_job_tick = jt;
        busy = false;
        ++done_count;
      }
    }
  }

  long long last_job_tick = -1;

  bool submit(const float* win, size_t n, long long tick_idx) {
    std::lock_guard<std::mutex> lk(mu);
    if (busy || has_job) {
      ++drop_count;
      return false;
    }
    memcpy(stage.data(), win, n * sizeof(float));
    job_tick = tick_idx;
    has_job = true;
    cv.notify_one();
    return true;
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv.notify_all();
    if (th.joinable()) th.join();
  }
};

static void usage() {  fprintf(stderr,
          "usage: sysbench2 [--backend bpu|onnx] [--hz 200] [--infer-every 1] [--duration 300]\n"
          "                 [--load 0] [--model PATH] [--data PATH] [--head PATH]\n"
          "                 [--threads 4] [--spin 0|1] [--rt PRIO] [--pin CPU]\n"
          "                 [--tag NAME] [--stats CSV] [--dump-all PATH]\n");
}

int main(int argc, char** argv) {
  std::string backend = "bpu";
  int hz = 200, infer_every = 1, duration = 300, load = 0;
  std::string model = "/home/lucas/bpu_spike/nature_bpu_realcal.bin";
  std::string data_path = "/home/lucas/data-transferrer/chunk_000/graph_inputs.bin";
  std::string head_path = "/home/lucas/data-transferrer/head_f32.bin";
  std::string tag = "run";
  std::string stats_csv;
  std::string dump_all;
  int threads = 4, spin = 1, async_mode = 0;
  int pin_cpu = -1, rt_prio = 0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--backend") backend = next();
    else if (a == "--hz") hz = atoi(next());
    else if (a == "--infer-every") infer_every = atoi(next());
    else if (a == "--duration") duration = atoi(next());
    else if (a == "--load") load = atoi(next());
    else if (a == "--model") model = next();
    else if (a == "--data") data_path = next();
    else if (a == "--head") head_path = next();
    else if (a == "--tag") tag = next();
    else if (a == "--stats") stats_csv = next();
    else if (a == "--dump-all") dump_all = next();
    else if (a == "--threads") threads = atoi(next());
    else if (a == "--spin") spin = atoi(next());
    else if (a == "--async") async_mode = 1;
    else if (a == "--pin") pin_cpu = atoi(next());
    else if (a == "--rt") rt_prio = atoi(next());
    else { usage(); return 2; }
  }
  if (infer_every < 1) infer_every = 1;
  const bool use_onnx = (backend == "onnx");
  const int HIST = use_onnx ? 249 : 249 + PAD;   // ONNX 图内含预处理([2,12,249])，BPU 图要补零到 264
  const size_t win_floats = (size_t)LEGS * CH * HIST;
  const size_t win_bytes = win_floats * sizeof(float);

  // ---------- 回放数据 ----------
  FILE* fd = fopen(data_path.c_str(), "rb");
  if (!fd) { fprintf(stderr, "cannot open data %s\n", data_path.c_str()); return 1; }
  fseek(fd, 0, SEEK_END);
  long dbytes = ftell(fd);
  fseek(fd, 0, SEEK_SET);
  size_t wins = (size_t)dbytes / win_bytes;
  if (wins == 0) { fprintf(stderr, "data too small (%ld < %zu)\n", dbytes, win_bytes); return 1; }
  std::vector<float> replay((size_t)wins * win_floats);
  if (fread(replay.data(), 1, wins * win_bytes, fd) != wins * win_bytes) {
    fprintf(stderr, "short read\n");
    return 1;
  }
  fclose(fd);

  // ---------- head 权重（仅 BPU 路径用：2x80 weight + 2 bias） ----------
  float headbuf[LEGS * OUT_CH + LEGS];
  memset(headbuf, 0, sizeof(headbuf));
  if (!use_onnx) {
    FILE* fh = fopen(head_path.c_str(), "rb");
    if (!fh) { fprintf(stderr, "cannot open head %s\n", head_path.c_str()); return 1; }
    if (fread(headbuf, sizeof(float), LEGS * OUT_CH + LEGS, fh) !=
        (size_t)(LEGS * OUT_CH + LEGS)) {
      fprintf(stderr, "bad head file\n");
      return 1;
    }
    fclose(fh);
  }

  // ---------- 后端初始化 ----------
  hbPackedDNNHandle_t packed = nullptr;
  hbDNNHandle_t dnn = nullptr;
  hbDNNTensor input, output;
  memset(&input, 0, sizeof(input));
  memset(&output, 0, sizeof(output));
  hbDNNInferCtrlParam ctrl;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);
  ctrl.bpuCoreId = 0;

  // onnxruntime
  Ort::Env* env = nullptr;
  Ort::Session* sess = nullptr;
  Ort::SessionOptions* so = nullptr;
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<float> ort_in;
  std::vector<int64_t> ort_shape;
  std::vector<const char*> in_names, out_names;
  std::vector<std::string> in_name_store, out_name_store;

  if (use_onnx) {
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "sysbench2");
    so = new Ort::SessionOptions();
    so->SetIntraOpNumThreads(threads);
    so->SetInterOpNumThreads(1);
    so->SetExecutionMode(ORT_SEQUENTIAL);
    so->SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    so->AddConfigEntry("session.intra_op.allow_spinning", spin ? "1" : "0");
    sess = new Ort::Session(*env, model.c_str(), *so);
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < sess->GetInputCount(); ++i) {
      in_name_store.push_back(sess->GetInputNameAllocated(i, alloc).get());
      auto shp = sess->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      for (int64_t d : shp) ort_shape.push_back(d);
    }
    for (size_t i = 0; i < sess->GetOutputCount(); ++i)
      out_name_store.push_back(sess->GetOutputNameAllocated(i, alloc).get());
    for (auto& s : in_name_store) in_names.push_back(s.c_str());
    for (auto& s : out_name_store) out_names.push_back(s.c_str());

    size_t expect = 1;
    for (int64_t d : ort_shape) expect *= (size_t)d;
    ort_in.assign(expect, 0.0f);
    fprintf(stderr, "[sysbench2] onnx=%s threads=%d spin=%d in_elems=%zu shape=", model.c_str(),
            threads, spin, expect);
    for (int64_t d : ort_shape) fprintf(stderr, "%lld ", (long long)d);
    fprintf(stderr, "\n");
    if (expect != win_floats)
      fprintf(stderr, "[sysbench2] WARN: model input elems %zu != window %zu\n", expect, win_floats);
  } else {
    const char* files[1] = {model.c_str()};
    if (hbDNNInitializeFromFiles(&packed, files, 1) != 0) {
      fprintf(stderr, "hbDNNInitializeFromFiles failed\n");
      return 1;
    }
    char const** names = nullptr;
    int32_t name_count = 0;
    hbDNNGetModelNameList(&names, &name_count, packed);
    if (hbDNNGetModelHandle(&dnn, packed, names[0]) != 0) {
      fprintf(stderr, "hbDNNGetModelHandle failed\n");
      return 1;
    }
    hbDNNGetInputTensorProperties(&input.properties, dnn, 0);
    if (hbSysAllocCachedMem(&input.sysMem[0], input.properties.alignedByteSize) != 0) {
      fprintf(stderr, "alloc input failed\n");
      return 1;
    }
    hbDNNGetOutputTensorProperties(&output.properties, dnn, 0);
    if (hbSysAllocCachedMem(&output.sysMem[0], output.properties.alignedByteSize) != 0) {
      fprintf(stderr, "alloc output failed\n");
      return 1;
    }
    fprintf(stderr, "[sysbench2] bpu=%s in_byte=%d out_byte=%d wins=%zu\n", names[0],
            input.properties.alignedByteSize, output.properties.alignedByteSize, wins);
  }

  // 单次推理：输入一个窗口，输出 4 个力矩
  auto infer_once = [&](const float* win, float out4[4]) -> int {
    double* acc = nullptr;
    (void)acc;
    if (use_onnx) {
      memcpy(ort_in.data(), win, win_floats * sizeof(float));
      Ort::Value tin = Ort::Value::CreateTensor<float>(mem_info, ort_in.data(), ort_in.size(),
                                                       ort_shape.data(), ort_shape.size());
      auto outs = sess->Run(Ort::RunOptions{nullptr}, in_names.data(), &tin, 1, out_names.data(), 1);
      const float* o = outs[0].GetTensorData<float>();
      for (int i = 0; i < 4; ++i) out4[i] = o[i];
      return 0;
    }
    memcpy(input.sysMem[0].virAddr, win, win_bytes);
    hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    hbDNNTaskHandle_t task = nullptr;
    hbDNNTensor* out_ptr = &output;
    int32_t ret = hbDNNInfer(&task, &out_ptr, &input, dnn, &ctrl);
    if (ret != 0 || task == nullptr) return -1;
    ret = hbDNNWaitTaskDone(task, 200);
    hbDNNReleaseTask(task);
    if (ret != 0) return -1;
    hbSysFlushMem(&output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    const float* feat = (const float*)output.sysMem[0].virAddr;
    for (int leg = 0; leg < LEGS; ++leg) {
      for (int j = 0; j < LEGS; ++j) {
        double a = headbuf[LEGS * OUT_CH + j];
        for (int k = 0; k < OUT_CH; ++k)
          a += (double)feat[((size_t)leg * OUT_CH + k) * OUT_LEN + (OUT_LEN - 1)] *
               headbuf[j * OUT_CH + k];
        out4[leg * 2 + j] = (float)a;
      }
    }
    return 0;
  };

  // ---------- 逐窗口输出（精度对拍） ----------
  if (!dump_all.empty()) {
    FILE* fo = fopen(dump_all.c_str(), "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", dump_all.c_str()); return 1; }
    std::vector<float> out4(4);
    long bad = 0;
    for (size_t w = 0; w < wins; ++w) {
      if (infer_once(replay.data() + w * win_floats, out4.data()) != 0) ++bad;
      fwrite(out4.data(), sizeof(float), 4, fo);
    }
    fclose(fo);
    printf("DUMP_ALL wins=%zu failed=%ld -> %s\n", wins, bad, dump_all.c_str());
    if (!use_onnx) { hbSysFreeMem(&input.sysMem[0]); hbSysFreeMem(&output.sysMem[0]); hbDNNRelease(packed); }
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
    if (pin_to(pin_cpu) == 0) fprintf(stderr, "[sysbench2] main thread pinned to cpu%d\n", pin_cpu);
    else fprintf(stderr, "[sysbench2] WARN: pin to cpu%d failed\n", pin_cpu);
  }

  std::vector<pid_t> kids;
  for (int i = 0; i < load; ++i) {
    pid_t p = fork();
    if (p == 0) {
      if (pin_cpu >= 0) pin_to(1 + (i % 7));
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

  // ---------- 环形窗口 ----------
  std::vector<float> ring(win_floats, 0.0f);
  memcpy(ring.data(), replay.data(), win_bytes);

  const double period = 1.0 / hz;
  const long total_ticks = (long)hz * duration;
  Stats jitter, work, infer_lat, age;
  long late = 0, late_hard = 0, over45 = 0, over75 = 0, infer_cnt = 0, infer_fail = 0;
  long long consec = 0, consec_max = 0;
  double worst_jitter = 0, worst_age = 0, worst_infer = 0;
  std::vector<double> tick_t0(infer_every, 0.0);

  jitter.reserve((size_t)total_ticks + 8);
  work.reserve((size_t)total_ticks + 8);
  infer_lat.reserve((size_t)total_ticks / infer_every + 8);
  age.reserve((size_t)total_ticks + 8);
  long spike_n = 0;

  ResMon mon;
  mon.begin(stats_csv.empty() ? nullptr : stats_csv.c_str());
  mon.win.reserve(512);

  if (rt_prio > 0) {
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = rt_prio;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0) {
      struct sched_param got;
      int pol = sched_getscheduler(0);
      sched_getparam(0, &got);
      fprintf(stderr, "[sysbench2] SCHED_FIFO prio=%d (policy=%d)\n", got.sched_priority, pol);
    } else {
      fprintf(stderr, "[sysbench2] WARN: SCHED_FIFO failed (%s)\n", strerror(errno));
    }
  }

  double base_cpu_s = proc_cpu_s();
  // 预热：跑 warm 次推理，不计入统计
  {
    std::vector<float> o4(4);
    for (int i = 0; i < 20; ++i) infer_once(replay.data() + (i % wins) * win_floats, o4.data());
  }
  double warm_cpu_s = proc_cpu_s();

  // 异步推理线程（真实部署形态：控制环路不等推理）
  std::vector<double> plan_time((size_t)total_ticks + 8, 0.0);
  long long frame_cursor = 0, last_seq = 0;
  AsyncInfer aio;
  if (async_mode) {
    aio.start(infer_once, win_floats);
    fprintf(stderr, "[sysbench2] async worker thread started\n");
  }

  double next = now_s() + period;
  double t_start = next;
  float torque[LEGS][2];
  memset(torque, 0, sizeof(torque));
  float out4[4] = {0, 0, 0, 0};

  for (long tick = 0; tick < total_ticks; ++tick) {
    next += period;
    struct timespec ts;
    ts.tv_sec = (time_t)next;
    ts.tv_nsec = (long)((next - (double)ts.tv_sec) * 1e9);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);

    double t0 = now_s();
    double late_ms = (t0 - next) * 1e3;
    jitter.add(late_ms);
    if (late_ms > 0.5) { ++late; ++consec; if (consec > consec_max) consec_max = consec; }
    else consec = 0;
    if (late_ms > period * 1e3) ++late_hard;
    if (late_ms > worst_jitter) worst_jitter = late_ms;
    if (late_ms > 5.0) {
      if (spike_n < 20)
        fprintf(stderr, "[spike] tick=%ld t=%.1fs jitter=%.1fms prev_work=%.1fms\n", tick,
                t0 - t_start, late_ms, work.v.empty() ? 0.0 : work.v.back());
      ++spike_n;
    }

    // 模拟 IMU：把回放窗口的最后一列当作"刚采到的新样本"
    const float* src = replay.data() + (size_t)(tick % wins) * win_floats;
    for (int leg = 0; leg < LEGS; ++leg) {
      float* row = ring.data() + (size_t)leg * CH * HIST;
      for (int c = 0; c < CH; ++c) {
        float* ch_row = row + (size_t)c * HIST;
        memmove(ch_row, ch_row + 1, (HIST - 1) * sizeof(float));
        ch_row[HIST - 1] = src[(size_t)leg * CH * HIST + (size_t)c * HIST + (HIST - 1)];
      }
    }

    tick_t0[tick % infer_every] = next;
    plan_time[(size_t)tick] = next;

    if (async_mode) {
      // 异步：主循环只投递窗口，不等待；结果就绪后在下一帧发布
      if (tick % infer_every == 0) aio.submit(ring.data(), win_floats, tick);
      long long seq = aio.done_count.load(std::memory_order_acquire);
      if (seq != last_seq) {
        double td, lat;
        long long jt;
        {
          std::lock_guard<std::mutex> lk(aio.mu);
          memcpy(out4, aio.result, sizeof(out4));
          td = aio.t_done;
          lat = aio.lat_ms;
          jt = aio.last_job_tick;
          last_seq = aio.done_count.load();
        }
        for (int leg = 0; leg < LEGS; ++leg) {
          torque[leg][0] = out4[leg * 2 + 0];
          torque[leg][1] = out4[leg * 2 + 1];
        }
        infer_lat.add(lat);
        mon.win.push_back(lat);
        if (lat > worst_infer) worst_infer = lat;
        ++infer_cnt;
        // 这批结果覆盖了从 frame_cursor 到 jt 的所有帧（含被跳过的窗口）
        for (long long idx = frame_cursor; idx <= jt; ++idx) {
          double a = (td - plan_time[(size_t)idx]) * 1e3;
          if (a < 0) continue;
          age.add(a);
          if (a > worst_age) worst_age = a;
          if (a > 45.0) ++over45;
          if (a > 75.0) ++over75;
        }
        if (jt + 1 > frame_cursor) frame_cursor = jt + 1;
      }
    } else if (tick % infer_every == 0) {
      double ti0 = now_s();
      int rc = infer_once(ring.data(), out4);
      double ti1 = now_s();
      if (rc != 0) {
        ++infer_fail;
      } else {
        for (int leg = 0; leg < LEGS; ++leg) {
          torque[leg][0] = out4[leg * 2 + 0];
          torque[leg][1] = out4[leg * 2 + 1];
        }
      }
      double lat = (ti1 - ti0) * 1e3;
      infer_lat.add(lat);
      mon.win.push_back(lat);
      if (lat > worst_infer) worst_infer = lat;
      ++infer_cnt;

      for (int b = 0; b < infer_every; ++b) {
        double tf = tick_t0[b];
        if (tf <= 0) continue;
        double a = (ti1 - tf) * 1e3;
        age.add(a);
        if (a > worst_age) worst_age = a;
        if (a > 45.0) ++over45;
        if (a > 75.0) ++over75;
      }
    }

    double w = (now_s() - t0) * 1e3;
    work.add(w);

    mon.tick(t0, infer_cnt, (tick > 0 && tick % (hz * 30) == 0));
  }

  double elapsed = now_s() - t_start;
  double run_cpu_s = proc_cpu_s() - warm_cpu_s;
  double warm_cpu_per_infer = (warm_cpu_s - base_cpu_s) / 20.0 * 1e3;
  mon.finish();

  jitter.sort(); work.sort(); infer_lat.sort(); age.sort();
  mon.proc_pct.sort(); mon.sys_pct.sort(); mon.core_max_pct.sort(); mon.t_cpu.sort();
  mon.t_ddr.sort(); mon.freq0.sort();

  double cpu_ms_per_infer = infer_cnt > 0 ? run_cpu_s * 1e3 / (double)infer_cnt : 0;

  printf("\n=== %s | %s | %d Hz loop, infer every %d (= %d Hz), %ds, load=%d, "
         "threads=%d spin=%d ===\n",
         tag.c_str(), backend.c_str(), hz, infer_every, hz / infer_every, duration, load,
         use_onnx ? threads : 0, use_onnx ? spin : 0);
  printf("actual rate      : %.3f Hz (%ld ticks in %.3f s)\n", (double)total_ticks / elapsed,
         total_ticks, elapsed);
  printf("late ticks       : %ld (%.4f%%) [>0.5ms]  hard misses %ld [>1 period]  "
         "max consecutive %lld\n",
         late, 100.0 * late / total_ticks, late_hard, consec_max);
  printf("tick jitter ms   : p50=%.3f p99=%.3f max=%.3f\n", jitter.pct(0.50), jitter.pct(0.99),
         jitter.max());
  printf("frame work ms    : p50=%.3f p99=%.3f max=%.3f\n", work.pct(0.50), work.pct(0.99),
         work.max());
  printf("infer latency ms : n=%ld p50=%.3f p95=%.3f p99=%.3f max=%.3f fail=%ld\n", infer_cnt,
         infer_lat.pct(0.50), infer_lat.pct(0.95), infer_lat.pct(0.99), worst_infer, infer_fail);
  printf("data->cmd ms     : n=%zu p50=%.3f p95=%.3f p99=%.3f max=%.3f\n", age.v.size(),
         age.pct(0.50), age.pct(0.95), age.pct(0.99), worst_age);
  printf("budget           : >45ms %ld (%.3f%%)  >75ms %ld (%.4f%%)\n", over45,
         100.0 * over45 / (double)(age.v.size() ? age.v.size() : 1), over75,
         100.0 * over75 / (double)(age.v.size() ? age.v.size() : 1));
  printf("sched spikes     : %ld ticks >5ms late\n", spike_n);
  if (async_mode)
    printf("async            : submitted %ld, dropped %ld (worker busy), loop never blocks\n",
           (long)(total_ticks / infer_every - aio.drop_count.load()), (long)aio.drop_count.load());
  printf("duty             : infer p99 %.1f%% of period, work p99 %.1f%%\n",
         100.0 * infer_lat.pct(0.99) / (period * 1e3),
         100.0 * work.pct(0.99) / (period * 1e3));
  printf("cpu cost/infer   : %.3f ms CPU per inference (run), %.3f ms (warm, no loop)\n",
         cpu_ms_per_infer, warm_cpu_per_infer);
  printf("proc cpu         : mean=%.1f%% p95=%.1f%% max=%.1f%% of one core\n", mon.proc_pct.mean(),
         mon.proc_pct.pct(0.95), mon.proc_pct.max());
  printf("sys cpu busy     : mean=%.1f%% max=%.1f%%  core-max mean=%.1f%% max=%.1f%%\n",
         mon.sys_pct.mean(), mon.sys_pct.max(), mon.core_max_pct.mean(), mon.core_max_pct.max());
  printf("temp C           : cpu max=%.1f mean=%.1f | ddr max=%.1f mean=%.1f\n",
         mon.t_cpu.max(), mon.t_cpu.mean(), mon.t_ddr.max(), mon.t_ddr.mean());
  printf("cpu0 freq MHz    : min=%.0f max=%.0f | rss start=%ldKB peak=%ldKB | mem avail min=%.0f MB\n",
         mon.freq_min > 1e8 ? 0 : mon.freq_min, mon.freq_max, mon.rss_start, mon.rss_peak,
         mon.mem_min_mb);

  printf("JSON {\"tag\":\"%s\",\"backend\":\"%s\",\"hz\":%d,\"infer_hz\":%d,\"duration\":%d,"
         "\"load\":%d,\"threads\":%d,\"spin\":%d,"
         "\"rate_hz\":%.3f,\"late\":%ld,\"late_pct\":%.5f,\"hard_miss\":%ld,"
         "\"consec_max\":%lld,"
         "\"jit_p50\":%.4f,\"jit_p99\":%.4f,\"jit_max\":%.4f,"
         "\"work_p99\":%.4f,\"work_max\":%.4f,"
         "\"infer_p50\":%.4f,\"infer_p95\":%.4f,\"infer_p99\":%.4f,\"infer_max\":%.4f,"
         "\"infer_fail\":%ld,"
         "\"age_p50\":%.4f,\"age_p95\":%.4f,\"age_p99\":%.4f,\"age_max\":%.4f,"
         "\"over45\":%ld,\"over75\":%ld,"
         "\"duty_infer_p99\":%.2f,\"cpu_ms_per_infer\":%.4f,\"warm_ms_per_infer\":%.4f,"
         "\"proc_cpu_mean\":%.2f,\"proc_cpu_p95\":%.2f,\"proc_cpu_max\":%.2f,"
         "\"sys_cpu_mean\":%.2f,\"sys_cpu_max\":%.2f,"
         "\"core_max_mean\":%.2f,\"core_max_max\":%.2f,"
         "\"t_cpu_max\":%.2f,\"t_cpu_mean\":%.2f,\"t_ddr_max\":%.2f,\"t_ddr_mean\":%.2f,"
         "\"freq_min\":%.0f,\"freq_max\":%.0f,"
         "\"rss_start_kb\":%ld,\"rss_peak_kb\":%ld,\"mem_avail_min_mb\":%.0f}\n",
         tag.c_str(), backend.c_str(), hz, hz / infer_every, duration, load,
         use_onnx ? threads : 0, use_onnx ? spin : 0,
         (double)total_ticks / elapsed, late, 100.0 * late / total_ticks, late_hard, consec_max,
         jitter.pct(0.50), jitter.pct(0.99), jitter.max(),
         work.pct(0.99), work.max(),
         infer_lat.pct(0.50), infer_lat.pct(0.95), infer_lat.pct(0.99), worst_infer, infer_fail,
         age.pct(0.50), age.pct(0.95), age.pct(0.99), worst_age, over45, over75,
         100.0 * infer_lat.pct(0.99) / (period * 1e3), cpu_ms_per_infer, warm_cpu_per_infer,
         mon.proc_pct.mean(), mon.proc_pct.pct(0.95), mon.proc_pct.max(),
         mon.sys_pct.mean(), mon.sys_pct.max(),
         mon.core_max_pct.mean(), mon.core_max_pct.max(),
         mon.t_cpu.max(), mon.t_cpu.mean(), mon.t_ddr.max(), mon.t_ddr.mean(),
         mon.freq_min > 1e8 ? 0 : mon.freq_min, mon.freq_max,
         mon.rss_start, mon.rss_peak, mon.mem_min_mb);

  if (async_mode) aio.shutdown();
  for (pid_t p : kids) waitpid(p, nullptr, 0);
  if (use_onnx) {
    delete sess;
    delete so;
    delete env;
  } else {
    hbSysFreeMem(&input.sysMem[0]);
    hbSysFreeMem(&output.sysMem[0]);
    hbDNNRelease(packed);
  }
  return 0;
}
