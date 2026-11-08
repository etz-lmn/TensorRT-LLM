/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Route tables of one adaptive EPLB layer, every expert's copies in table order.
struct AdaptiveEplbRouteTables
{
    int const* counts; //!< [numExperts] copies of every expert
    int const* starts; //!< [numExperts] table index of every expert's first copy
    int const* slots;  //!< [numSlots] slot of every table entry
    //! [numSlots] for an expert with several copies, the share of its copies up to entry k, in units of 2^-32
    uint32_t const* cumulativeShares;
    int numExperts;
    int numSlots;
};

//! An MoE backend's staging buffers of a bucket of total / topK token rows, which the route writes into. The routed
//! elements' slots and gate weights fill the first rows; the rows after them get slot -1 and weight 0. Without
//! staging, weightsOut is null and total equals the number of routed elements.
struct AdaptiveEplbStaging
{
    float const* weightsIn; //!< [count] gate weights of the routed elements
    float* weightsOut;      //!< [total / topK, ldWeights] rows, the first topK columns written
    int64_t ldWeights;
    int topK;
    int64_t total; //!< elements of the bucket (the out buffer holds that many), >= count
};

//! A layer's statistics, which the route's launch can update as adaptiveEplbRecord does. Without statistics, ema is
//! null.
struct AdaptiveEplbStats
{
    float* ema;
    int64_t* forwards;
    int const* enabled;
    float decay;
    bool first;
};

//! Route the flat [count] expert ids of one EP rank to slots.
//!
//! An expert with one copy takes its slot. The elements of an expert e with c > 1 copies are spread over its copies in
//! proportion to their shares, separately in each split block of 1024 consecutive elements: in split block b, the
//! element that is the j-th of the n elements of e there takes the first copy k < c - 1 with
//! u < cumulativeShares[starts[e] + k], else copy c - 1, where
//!     u = e * kExpertMultiplier + (epRank * kBlocksPerRank + b) * kBlockMultiplier + j * floor(2^32 / n)   (mod 2^32)
//! (2^32 - 1 replaces floor(2^32 / n) for n = 1): n evenly spaced points from an offset that differs between experts,
//! ranks and split blocks. An id outside the experts, or a table entry outside the slots, takes numSlots. With staging,
//! the launch also writes the gate weights and the staging rows after the routed elements; with statistics, one more
//! block of the launch updates them as adaptiveEplbRecord does.
template <typename IdT, typename OutT>
void adaptiveEplbRoute(IdT const* ids, int64_t count, int epRank, AdaptiveEplbRouteTables tables, OutT* out,
    AdaptiveEplbStaging staging, AdaptiveEplbStats stats, cudaStream_t stream);

//! Add a forward's expert ids to a layer's statistics.
//!
//! Nothing happens while enabled[0] == 0 (read on the device, so a captured graph records only while it is set).
//! Otherwise, with n[e] the number of ids equal to e (ids outside [0, numExperts) are not counted), the first call of
//! a forward sets ema[e] = fl32(fl32(ema[e] * decay) + n[e]), rounding after the multiply and after the add, and
//! increments forwards[0]; a later call of the same forward (a further chunk or repeat) adds n[e] to ema[e]. One block
//! counts the ids.
template <typename IdT>
void adaptiveEplbRecord(IdT const* ids, int64_t count, int numExperts, float* ema, int64_t* forwards,
    int const* enabled, float decay, bool first, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
