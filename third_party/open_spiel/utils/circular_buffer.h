// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_OPEN_SPIEL_UTILS_CIRCULAR_BUFFER_H_
#define THIRD_PARTY_OPEN_SPIEL_UTILS_CIRCULAR_BUFFER_H_

#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

namespace open_spiel {

// A simple circular buffer of fixed size.
template <class T>
class CircularBuffer {
 public:
  explicit CircularBuffer(int max_size)
      : max_size_(max_size), total_added_(0) {}

  // Add one element, replacing the oldest once it's full.
  virtual void Add(const T& value) {
    if (data_.size() < max_size_) {
      data_.push_back(value);
    } else {
      data_[total_added_ % max_size_] = value;
    }
    total_added_ += 1;
  }

  virtual void Add(const T& value, std::mt19937& gen) { Add(value); }

  std::unordered_set<size_t> floyd_sample(std::mt19937& gen, const size_t& k,
                                          const size_t& N) {
    std::unordered_set<size_t> elems(k);  // preallocation is good
    for (size_t r = N - k; r < N; ++r) {
      size_t v = std::uniform_int_distribution<>(1, r)(gen);
      if (!elems.insert(v).second) elems.insert(r);
    }
    return elems;
  }

  // Return `num` elements without replacement.
  // Thread-safe.
  virtual std::vector<T> Sample(std::mt19937& gen, int num) {
    int sample_num = std::min(num, (int)data_.size());
    if (sample_num == data_.size()) {
      return data_;
    }
    std::unordered_set<size_t> indexes =
        floyd_sample(gen, sample_num, data_.size());
    std::vector<T> out(sample_num);
    if (sample_num <= 0) {
      return out;
    }
    int i = 0;
    for (auto ind : indexes) {
      out[i] = data_[ind];
      ++i;
    }
    return out;
  }

  // Return the full buffer.
  const std::vector<T>& Data() const { return data_; }

  // Access a single element from the buffer.
  const T& operator[](int i) const { return data_[i]; }

  // How many elements are in the buffer.
  int Size() const { return data_.size(); }

  // Is the buffer empty?
  bool Empty() const { return data_.empty(); }

  // How many elements have ever been added to the buffer.
  int64_t TotalAdded() const { return total_added_; }

  virtual void Clear() {
    data_.clear();
    total_added_ = 0;
  }

 protected:
  const int max_size_;
  int64_t total_added_;
  std::uniform_int_distribution<int64_t> dis_;
  std::vector<T> data_;
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_UTILS_CIRCULAR_BUFFER_H_
