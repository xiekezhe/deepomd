#include <omp.h>

#include <array>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "open_spiel/utils/thread_pool.h"

template <typename TimeT = std::chrono::milliseconds>
struct measure {
  template <typename F, typename... Args>
  static TimeT duration(F&& func, Args&&... args) {
    auto start = std::chrono::steady_clock::now();
    std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
    return std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() -
                                             start);
  }
};

std::vector<int> task() {
  std::vector<int> test{100};
  for (int j = 0; j != 1000; ++j) {
    std::vector<int> test1;
    for (int i = 10; i != 1000; ++i) {
      test1.push_back(i * i);
    }
  }
  return test;
}

int master(int num_threads = 1) {
  int num_tasks = 200;
  int sum = 0;
  int over_ratio = 2;
  if (num_threads == 0) {
    for (int t = 1; t != num_tasks + 1; ++t) {
      auto value = task();
      sum += value[0];
    }
    return sum;
  }
#pragma omp parallel
  {
#pragma omp for reduction(+ : sum) schedule(dynamic)
    for (int t = 1; t != num_tasks + 1; ++t) {
      auto result = task();
      sum += result[0];
    }
  }
  return sum;
}

int main() {
  for (int nt = 0; nt <= 10; ++nt) {
    omp_set_num_threads(nt);
    int milli_time = measure<>::duration([nt]() {
                       for (int i = 0; i != 10; ++i) master(nt);
                     }).count();
    std::cout << "thread_num = " << nt << ": " << master(nt) << " "
              << milli_time << "ms" << std::endl;
  }
}