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
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Stages one MegaMoE launch's routes into its bucket in one launch: rows [0, liveRows) of the [rows, topK] buffers get
//! the routes' expert ids (converted to OutT) and float32 gate weights, rows [liveRows, rows) get id -1 and weight 0.
//! weightsOut has a unit column stride and row stride weightsRowStride; its columns from topK on are not written.
template <typename IdT, typename OutT>
void megaMoeStageRoutes(IdT const* ids, float const* weights, int64_t liveRows, OutT* idsOut, float* weightsOut,
    int64_t weightsRowStride, int64_t rows, int topK, cudaStream_t stream);

//! The NVFP4 quantization of a MegaMoE launch's activation into its bucket staging: [rows, cols] bfloat16 in, scaled
//! by globalScale[0], to [rows, cols / 2] bytes of E2M1 pairs (out) and [rows, cols / 16] UE4M3 scale factors (sfOut,
//! the linear layout), element for element as fp4_quantize_into writes them. cols is a multiple of 16.
struct MegaMoeQuantize
{
    __nv_bfloat16 const* in;
    float const* globalScale;
    uint32_t* out;
    uint8_t* sfOut;
    int rows;
    int cols;
};

//! megaMoeStageRoutes and the quantization of the launch's activation in one launch, whose blocks run both at once.
template <typename IdT, typename OutT>
void megaMoeStageInputs(MegaMoeQuantize const& quantize, IdT const* ids, float const* weights, int64_t liveRows,
    OutT* idsOut, float* weightsOut, int64_t weightsRowStride, int64_t rows, int topK, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
