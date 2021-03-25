#include "paddle/fluid/framework/stats.h"
namespace stats {

Stats::Stats() {}

BenchmarkStats* Stats::benchmark_stats = nullptr;

BenchmarkStats* Stats::getBenchmarkStats() {
  if (nullptr == benchmark_stats) {
    benchmark_stats = new BenchmarkStats("Stats", 100);
  }
  return benchmark_stats;
}

}
