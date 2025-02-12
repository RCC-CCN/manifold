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

#include "./iters.h"

namespace manifold {

// `scatter` copies elements from a source range into an output array according
// to a map. For each iterator `i` in the range `[first, last)`, the value `*i`
// is assigned to `outputFirst[mapFirst[i - first]]`.  If the same index appears
// more than once in the range `[mapFirst, mapFirst + (last - first))`, the
// result is undefined.
//
// The map range, input range and the output range must not overlap.
template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator>
void scatter(InputIterator1 first, InputIterator1 last, InputIterator2 mapFirst,
             OutputIterator outputFirst) {
  std::for_each(countAt(0),
                countAt(static_cast<size_t>(std::distance(first, last))),
                [first, mapFirst, outputFirst](size_t i) {
                  outputFirst[mapFirst[i]] = first[i];
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
  std::for_each(countAt(0),
                countAt(static_cast<size_t>(std::distance(mapFirst, mapLast))),
                [mapFirst, inputFirst, outputFirst](size_t i) {
                  outputFirst[i] = inputFirst[mapFirst[i]];
                });
}

// Write `[0, last - first)` to the range `[first, last)`.
template <typename Iterator>
void sequence(Iterator first, Iterator last) {
  std::for_each(countAt(0),
                countAt(static_cast<size_t>(std::distance(first, last))),
                [first](size_t i) { first[i] = i; });
}

}  // namespace manifold
