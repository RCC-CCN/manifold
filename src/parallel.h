// Copyright 2022 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Simple implementation of selected functions in PSTL.
// Iterators must be RandomAccessIterator.

#pragma once

#include <algorithm>
#include <numeric>

#include "./iters.h"

namespace manifold {

enum class ExecutionPolicy {
  Par,
  Seq,
};

constexpr size_t kSeqThreshold = 1e4;
// ExecutionPolicy:
// - Sequential for small workload,
// - Parallel (CPU) for medium workload,
inline constexpr ExecutionPolicy autoPolicy(size_t size,
                                            size_t threshold = kSeqThreshold) {
  if (size <= threshold) {
    return ExecutionPolicy::Seq;
  }
  return ExecutionPolicy::Par;
}

template <typename Iter,
          typename Dummy = std::enable_if_t<!std::is_integral_v<Iter>>>
inline constexpr ExecutionPolicy autoPolicy(Iter first, Iter last,
                                            size_t threshold = kSeqThreshold) {
  if (static_cast<size_t>(std::distance(first, last)) <= threshold) {
    return ExecutionPolicy::Seq;
  }
  return ExecutionPolicy::Par;
}

template <typename InputIter, typename OutputIter>
void copy(ExecutionPolicy policy, InputIter first, InputIter last,
          OutputIter d_first);
template <typename InputIter, typename OutputIter>
void copy(InputIter first, InputIter last, OutputIter d_first);

// Applies the function `f` to each element in the range `[first, last)`
template <typename Iter, typename F>
void for_each(ExecutionPolicy policy, Iter first, Iter last, F f) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<Iter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");

  std::for_each(first, last, f);
}

// Applies the function `f` to each element in the range `[first, last)`
template <typename Iter, typename F>
void for_each_n(ExecutionPolicy policy, Iter first, size_t n, F f) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<Iter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  for_each(policy, first, first + n, f);
}

// Transform and reduce the range `[first, last)` by first applying a unary
// function `g`, and then combining the results using a binary operation `f`
// with an initial value `init`.
//
// The binary operation should be commutative and associative. Otherwise, the
// result is non-deterministic.
template <typename InputIter, typename BinaryOp, typename UnaryOp,
          typename T = std::invoke_result_t<
              UnaryOp, typename std::iterator_traits<InputIter>::value_type>>
T transform_reduce(ExecutionPolicy policy, InputIter first, InputIter last,
                   T init, BinaryOp f, UnaryOp g) {
  return std::reduce(policy, TransformIterator(first, g),
                     TransformIterator(last, g), init, f);
}

// Transform and reduce the range `[first, last)` by first applying a unary
// function `g`, and then combining the results using a binary operation `f`
// with an initial value `init`.
//
// The binary operation should be commutative and associative. Otherwise, the
// result is non-deterministic.
template <typename InputIter, typename BinaryOp, typename UnaryOp,
          typename T = std::invoke_result_t<
              UnaryOp, typename std::iterator_traits<InputIter>::value_type>>
T transform_reduce(InputIter first, InputIter last, T init, BinaryOp f,
                   UnaryOp g) {
  return std::reduce(TransformIterator(first, g), TransformIterator(last, g),
                     init, f);
}

// Copy the input range `[first, last)` to the output range
// starting from `d_first`.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must not overlap.
template <typename InputIter, typename OutputIter>
void copy(ExecutionPolicy policy, InputIter first, InputIter last,
          OutputIter d_first) {
  static_assert(std::is_convertible_v<
                    typename std::iterator_traits<InputIter>::iterator_category,
                    std::random_access_iterator_tag>,
                "You can only parallelize RandomAccessIterator.");
  static_assert(
      std::is_convertible_v<
          typename std::iterator_traits<OutputIter>::iterator_category,
          std::random_access_iterator_tag>,
      "You can only parallelize RandomAccessIterator.");

  std::copy(first, last, d_first);
}

// Copy the input range `[first, last)` to the output range
// starting from `d_first`.
//
// The input range `[first, last)` and
// the output range `[d_first, d_first + last - first)`
// must not overlap.
template <typename InputIter, typename OutputIter>
void copy(InputIter first, InputIter last, OutputIter d_first) {
  copy(autoPolicy(first, last, 1e6), first, last, d_first);
}

// Copy the input range `[first, first + n)` to the output range
// starting from `d_first`.
//
// The input range `[first, first + n)` and
// the output range `[d_first, d_first + n)`
// must not overlap.
template <typename InputIter, typename OutputIter>
void copy_n(ExecutionPolicy policy, InputIter first, size_t n,
            OutputIter d_first) {
  copy(policy, first, first + n, d_first);
}

// Copy the input range `[first, first + n)` to the output range
// starting from `d_first`.
//
// The input range `[first, first + n)` and
// the output range `[d_first, d_first + n)`
// must not overlap.
template <typename InputIter, typename OutputIter>
void copy_n(InputIter first, size_t n, OutputIter d_first) {
  copy(autoPolicy(n, 1e6), first, first + n, d_first);
}

// `scatter` copies elements from a source range into an output array according
// to a map. For each iterator `i` in the range `[first, last)`, the value `*i`
// is assigned to `outputFirst[mapFirst[i - first]]`.  If the same index appears
// more than once in the range `[mapFirst, mapFirst + (last - first))`, the
// result is undefined.
//
// The map range, input range and the output range must not overlap.
template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
void scatter(ExecutionPolicy policy, InputIterator1 first, InputIterator1 last,
             InputIterator2 mapFirst, OutputIterator outputFirst) {
  for_each(policy, countAt(0),
           countAt(static_cast<size_t>(std::distance(first, last))),
           [first, mapFirst, outputFirst](size_t i) {
             outputFirst[mapFirst[i]] = first[i];
           });
}

// `scatter` copies elements from a source range into an output array according
// to a map. For each iterator `i` in the range `[first, last)`, the value `*i`
// is assigned to `outputFirst[mapFirst[i - first]]`. If the same index appears
// more than once in the range `[mapFirst, mapFirst + (last - first))`,
// the result is undefined.
//
// The map range, input range and the output range must not overlap.
template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
void scatter(InputIterator1 first, InputIterator1 last, InputIterator2 mapFirst,
             OutputIterator outputFirst) {
  scatter(autoPolicy(first, last, 1e5), first, last, mapFirst, outputFirst);
}

// `gather` copies elements from a source array into a destination range
// according to a map. For each input iterator `i`
// in the range `[mapFirst, mapLast)`, the value `inputFirst[*i]`
// is assigned to `outputFirst[i - map_first]`.
//
// The map range, input range and the output range must not overlap.
template <typename InputIterator, typename RandomAccessIterator,
          typename OutputIterator>
void gather(ExecutionPolicy policy, InputIterator mapFirst,
            InputIterator mapLast, RandomAccessIterator inputFirst,
            OutputIterator outputFirst) {
  for_each(policy, countAt(0),
           countAt(static_cast<size_t>(std::distance(mapFirst, mapLast))),
           [mapFirst, inputFirst, outputFirst](size_t i) {
             outputFirst[i] = inputFirst[mapFirst[i]];
           });
}

// `gather` copies elements from a source array into a destination range
// according to a map. For each input iterator `i`
// in the range `[mapFirst, mapLast)`, the value `inputFirst[*i]`
// is assigned to `outputFirst[i - map_first]`.
//
// The map range, input range and the output range must not overlap.
template <typename InputIterator, typename RandomAccessIterator,
          typename OutputIterator>
void gather(InputIterator mapFirst, InputIterator mapLast,
            RandomAccessIterator inputFirst, OutputIterator outputFirst) {
  gather(autoPolicy(std::distance(mapFirst, mapLast), 1e5), mapFirst, mapLast,
         inputFirst, outputFirst);
}

// Write `[0, last - first)` to the range `[first, last)`.
template <typename Iterator>
void sequence(ExecutionPolicy policy, Iterator first, Iterator last) {
  for_each(policy, countAt(0),
           countAt(static_cast<size_t>(std::distance(first, last))),
           [first](size_t i) { first[i] = i; });
}

// Write `[0, last - first)` to the range `[first, last)`.
template <typename Iterator>
void sequence(Iterator first, Iterator last) {
  sequence(autoPolicy(first, last, 1e5), first, last);
}

}  // namespace manifold
