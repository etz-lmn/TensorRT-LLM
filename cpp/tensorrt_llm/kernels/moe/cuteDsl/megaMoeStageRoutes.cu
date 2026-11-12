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

#include "tensorrt_llm/kernels/moe/cuteDsl/megaMoeStageInputs.cuh"
#include "tensorrt_llm/kernels/moe/cuteDsl/megaMoeStageRoutes.h"

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int64_t kMaxBlocks = 1024; // a larger bucket loops: one launch either way

template <typename IdT, typename OutT>
__global__ void megaMoeStageRoutesKernel(MegaMoeStageWriter<IdT, OutT> writer)
{
    writer(static_cast<int>(blockIdx.x));
}

template <typename IdT, typename OutT>
MegaMoeStageWriter<IdT, OutT> stageWriter(IdT const* ids, float const* weights, int64_t n, int32_t const* liveRowWord,
    OutT* idsOut, float* weightsOut, int64_t rows, int topK)
{
    int64_t const elements = rows * topK;
    auto const blocks
        = static_cast<int>(std::min((elements + kMegaMoeStageThreads - 1) / kMegaMoeStageThreads, kMaxBlocks));
    return {ids, weights, n, liveRowWord, topK, idsOut, weightsOut, elements, blocks};
}

} // namespace

template <typename IdT, typename OutT>
void megaMoeStageRoutes(IdT const* ids, float const* weights, int64_t n, int32_t const* liveRowWord, OutT* idsOut,
    float* weightsOut, int64_t rows, int topK, cudaStream_t stream)
{
    auto const writer = stageWriter(ids, weights, n, liveRowWord, idsOut, weightsOut, rows, topK);
    if (writer.blocks > 0)
    {
        megaMoeStageRoutesKernel<IdT, OutT><<<writer.blocks, kMegaMoeStageThreads, 0, stream>>>(writer);
    }
}

template <typename IdT, typename OutT>
void megaMoeStageInputs(MegaMoeQuantize const& quantize, IdT const* ids, float const* weights, int64_t n,
    int32_t const* liveRowWord, OutT* idsOut, float* weightsOut, int64_t rows, int topK, cudaStream_t stream)
{
    launchMegaMoeStageInputs(
        quantize, stageWriter(ids, weights, n, liveRowWord, idsOut, weightsOut, rows, topK), 0, stream);
}

#define INSTANTIATE_MEGAMOE_STAGE_ROUTES(IdT, OutT)                                                                    \
    template void megaMoeStageRoutes<IdT, OutT>(                                                                       \
        IdT const*, float const*, int64_t, int32_t const*, OutT*, float*, int64_t, int, cudaStream_t);                 \
    template void megaMoeStageInputs<IdT, OutT>(MegaMoeQuantize const&, IdT const*, float const*, int64_t,             \
        int32_t const*, OutT*, float*, int64_t, int, cudaStream_t)

INSTANTIATE_MEGAMOE_STAGE_ROUTES(int32_t, int32_t);
INSTANTIATE_MEGAMOE_STAGE_ROUTES(int32_t, int64_t);
INSTANTIATE_MEGAMOE_STAGE_ROUTES(int64_t, int32_t);
INSTANTIATE_MEGAMOE_STAGE_ROUTES(int64_t, int64_t);

#undef INSTANTIATE_MEGAMOE_STAGE_ROUTES

} // namespace kernels

TRTLLM_NAMESPACE_END
