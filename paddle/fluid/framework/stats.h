#pragma once

#include <vector>
#include <cassert>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>

namespace stats {

class BenchmarkStats {
 public:
  BenchmarkStats(std::string name, int num_warmup_runs)
      : name_{name}, num_warmup_runs_{num_warmup_runs}{}

  void StartRun() {
    ++cur_count_;
    // Start recording CPU time.
    cur_start_walltime_ = std::chrono::steady_clock::now();
    cur_start_cpu_      = std::clock();
  }

  void StopRun() {
    // Do not collect the runtime statistics if we are still in the warm up
    // period.
    if (cur_count_ <= num_warmup_runs_) return;

    // Stop the CPU timer.
    std::clock_t cur_stop_cpu_ = std::clock();

    // Stop the wall clock timer.
    auto cur_stop_walltime_ = std::chrono::steady_clock::now();

    // Collect the wall clock duration.
    auto duration_walltime_ = cur_stop_walltime_ - cur_start_walltime_;
    run_times_walltime_.push_back(duration_walltime_);

    // Collect the CPU duration in microseconds.
    // First cast to integer that represents microseconds with truncation, as
    // does std::chrono::duration_cast. Then cast to std::chrono::microseconds.
    std::clock_t duration_cpu_raw = cur_stop_cpu_ - cur_start_cpu_;
    auto duration_cpu_ =
        static_cast<std::chrono::nanoseconds>(static_cast<int64_t>(1e9 * duration_cpu_raw / CLOCKS_PER_SEC));

    run_times_cpu_.push_back(duration_cpu_);

    total_duration_walltime_ += duration_walltime_;
    total_duration_cpu_ += duration_cpu_;
  }

  // Summarize the benchmark results.
  void Summarize() {
    std::sort(run_times_walltime_.begin(), run_times_walltime_.end());
    std::sort(run_times_cpu_.begin(), run_times_cpu_.end());

    auto percentile = [](double p, const std::vector<std::chrono::nanoseconds> &run_times) {
      assert(p >= 0.0 && p <= 1.0);
      return run_times[run_times.size() * p];
    };

    // BM: prefix is added to make grepping results from lit output easier.
    std::string prefix;
    prefix = "BM:" + name_ + ":";
    auto cpu_utilization = total_duration_cpu_.count() * 100.0 / total_duration_walltime_.count();

    std::cout << prefix << "Count: " << run_times_walltime_.size() << '\n';
    std::cout << prefix << "Duration(ns): " << total_duration_walltime_.count() << '\n';
    std::cout << prefix << "Time Min(ns): " << run_times_walltime_.front().count() << '\n';
    std::cout << prefix << "Time Max(ns): " << run_times_walltime_.back().count() << '\n';
    std::cout << prefix << "Time 50%(ns): " << percentile(0.5, run_times_walltime_).count() << '\n';
    std::cout << prefix << "Time 95%(ns): " << percentile(0.95, run_times_walltime_).count() << '\n';
    std::cout << prefix << "Time 99%(ns): " << percentile(0.99, run_times_walltime_).count() << '\n';
    // Log CPU time statistics.
    std::cout << prefix << "CPU Duration(ns): " << total_duration_cpu_.count() << '\n';
    std::cout << prefix << "CPU Min(ns): " << run_times_cpu_.front().count() << '\n';
    std::cout << prefix << "CPU Max(ns): " << run_times_cpu_.back().count() << '\n';
    std::cout << prefix << "CPU 50%(ns): " << percentile(0.5, run_times_cpu_).count() << '\n';
    std::cout << prefix << "CPU 95%(ns): " << percentile(0.95, run_times_cpu_).count() << '\n';
    std::cout << prefix << "CPU 99%(ns): " << percentile(0.99, run_times_cpu_).count() << '\n';
    std::cout << prefix << "CPU utilization(percent): " << cpu_utilization << "\n";
    std::cout << std::flush;
  }

 private:
  const std::string name_;
  const int num_warmup_runs_;
  int cur_count_ = 0;
  std::chrono::nanoseconds total_duration_walltime_{};
  std::chrono::nanoseconds total_duration_cpu_{};
  std::chrono::time_point<std::chrono::steady_clock> cur_start_walltime_{};
  std::clock_t cur_start_cpu_;
  std::vector<std::chrono::nanoseconds> run_times_walltime_;
  // CPU run times in microseconds.
  std::vector<std::chrono::nanoseconds> run_times_cpu_;
};

class Stats {
 private:
  static BenchmarkStats *benchmark_stats;
  Stats();

 public:
  static BenchmarkStats *getBenchmarkStats();
};  // class Stats


} // namespace stats
