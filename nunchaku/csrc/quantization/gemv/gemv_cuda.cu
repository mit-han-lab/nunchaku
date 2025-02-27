/*
 * Modified from NVIDIA [TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/d37b507f41a87457fe9f10f7459d08f5db235745/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv)
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
*/

#include <stdio.h>
#include <torch/extension.h>
#include "gemv_cuda.h"
#include "../dequantize.cuh"
#include "../../utils.cuh"

#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128

// Reduce sum within the warp using the tree reduction algorithm.
template <typename fp_t, int Num, int WarpSize>
__device__ __forceinline__ static void warp_reduce(fp_t *psum, float (*out_smem)[Num * 4])
{
  // kInterleave = 4
  float fpsum[Num];
#pragma unroll
  for (int i = 0; i < Num; ++i)
  {
    fpsum[i] = static_cast<float>(psum[i]);
  }

#pragma unroll
  for (int i = 0; i < Num; ++i)
  {
    // T0 + T1 + T8 + T9 + T16 + T17 + T24 + T25 (kInterleave = 4)
    fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 16);
    fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 8);
    fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 1);
  }
  __syncthreads();
  int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
  if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
  {
#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
      out_smem[warp][i * 4 + lane / 2] = fpsum[i];
    }
  }
  __syncthreads();
};

__device__ __forceinline__ int make_divisible(int c, int divisor)
{
  return (c + divisor - 1) / divisor;
}

template <typename f16_t, int NPerBlock, int Batch, int BlockSize, int GroupSize>
__global__ void gemv_kernel(
    const f16_t *inputs, const uint32_t *weight, const f16_t *scales, const f16_t *zeros, f16_t *outputs,
    const int IC, const int OC)
{

  using f162_t = typename packed_as<f16_t, 2>::type;
  using accum_t = float;
  using accum2_t = typename packed_as<accum_t, 2>::type;

  const int kStride = 64;
  const int kElemsPerThread = MEM_ACCESS_SIZE / 4;
  const int kThreadsNumPerTile = kStride / kElemsPerThread;

  static constexpr int kShuffleBasicTile = 2;
  static constexpr int kShuffleContinous = 4;
  static constexpr int kShuffleStrided = 4;

  constexpr int Num = NPerBlock * Batch;
  constexpr int kInterleave = 4;

  alignas(16) f16_t local_inputs[kElemsPerThread];
  alignas(16) uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
  alignas(16) f16_t half_weight_buffer[kElemsPerThread];
  alignas(16) f16_t dequantized_weight[kElemsPerThread * NPerBlock];
  alignas(16) f16_t local_scale[NPerBlock];
  alignas(16) f16_t local_scaled_zeros[NPerBlock];

  accum_t psum[Num];
  for (int i = 0; i < Num; ++i)
    psum[i] = static_cast<accum_t>(0.f);

  extern __shared__ uint8_t shmem[];
  float(*out_smem)[Num * kInterleave] = reinterpret_cast<float(*)[Num * kInterleave]>(shmem);

  const int blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
  const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
  const int act_k_offset = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
  const int group_offset = act_k_offset / GroupSize;
  // TODO: use make_divisible
  const uint32_t *blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;
  const f16_t *scale_ptr = scales + blk_row_offset + thd_row_offset + group_offset * OC;
  const f16_t *zeros_ptr = zeros + blk_row_offset + thd_row_offset + group_offset * OC;
  const f16_t *inputs_ptr = inputs + act_k_offset;

  const int act_forward_step = BlockSize * kElemsPerThread / kInterleave;
  const int scale_forward_step = act_forward_step / GroupSize * OC;

  // Main loop iteration, each block completes the outputs for several OCs
  for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
  {
// Load qweight, scales and scaled_zeros
#pragma unroll
    for (int idx = 0; idx < NPerBlock; ++idx)
    {
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4, 128 bit)
      *((float4 *)(local_qweights)) =
          *((float4 *)(blk_weight_ptr + (idx * kInterleave * IC + kk) / PACK_FACTOR));
      local_scale[idx] = *(scale_ptr + idx * kInterleave);
      local_scaled_zeros[idx] = *(zeros_ptr + idx * kInterleave);

// Map int4 qweight to fp format
#pragma unroll
      for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
      {
        // Converts 32 bits (8 x int4) to 8 fp16
        dequantize_s4_to_f16x2(*reinterpret_cast<f162_t *>(local_qweights + i), reinterpret_cast<uint4 *>(half_weight_buffer + i * PACK_FACTOR));
      }

// Dequantize (apply s/z) and shuffle elements to match the weight packing format
#pragma unroll
      for (int i = 0; i < kShuffleContinous; ++i)
      {
#pragma unroll
        for (int j = 0; j < kShuffleStrided; ++j)
        {
          f162_t w =
              *reinterpret_cast<f162_t *>(
                  half_weight_buffer + (i + j * kShuffleContinous) * kShuffleBasicTile);
          w = __hfma2(w, f162f162(local_scale[idx]), f162f162(local_scaled_zeros[idx]));
          dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0) * NPerBlock + idx] = w.x;
          dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1) * NPerBlock + idx] = w.y;
        }
      }
    }
#pragma unroll
    for (int batch_idx = 0; batch_idx < Batch; ++batch_idx)
    {
      const f16_t *local_inputs_ptr = inputs_ptr + batch_idx * IC;
#pragma unroll
      for (int idx = 0; idx < kElemsPerThread / 8; ++idx)
      {
        // load activation, 8 halves (128 bits) / step.
        *((float4 *)(local_inputs + idx * 8)) = *((float4 *)(local_inputs_ptr + idx * 8));
      }
// Perform the MACs
#pragma unroll
      for (int x = 0; x < NPerBlock / 2; ++x)
      {
#pragma unroll
        for (int y = 0; y < kElemsPerThread; ++y)
        {
          accum2_t prod = cuda_cast<accum2_t>(__hmul2(
              *reinterpret_cast<f162_t *>(dequantized_weight + y * NPerBlock + x * 2),
              f162f162(local_inputs[y])));
          *reinterpret_cast<accum2_t *>(psum + batch_idx * NPerBlock + x * 2) = prod + *reinterpret_cast<accum2_t *>(psum + batch_idx * NPerBlock + x * 2);
        }
      }
    }
    inputs_ptr += act_forward_step;
    scale_ptr += scale_forward_step;
    zeros_ptr += scale_forward_step;
  }

  warp_reduce<accum_t, Num, WARP_SIZE>(psum, out_smem);

  // Num * Interleave = batch * NPerBlock * Interleave -> 1 thread_block write back num
  for (int i = threadIdx.x; i < Num * kInterleave; i += BlockSize)
  {
    int batch_idx = i / (NPerBlock * kInterleave);
    int oc_idx = i % (NPerBlock * kInterleave);
    float acc = 0.f;
    for (int j = 0; j < BlockSize / WARP_SIZE; ++j)
    {
      acc += out_smem[j][i];
    }
    outputs[batch_idx * OC + blk_row_offset + oc_idx] = static_cast<f16_t>(acc);
  }
}

/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC, IC // 8];
  _zeros: int tensor of shape [OC, IC // G // 8];
  _scaling_factors: tensor of shape [OC, IC // G];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;

Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor awq_gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int m,
    int n,
    int k,
    int group_size)
{
  std::vector<int64_t> output_shape = _in_feats.sizes().vec();
  output_shape.back() = n;

  auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out_feats = torch::empty(output_shape, options);

  static constexpr int N_PER_BLOCK = 2;
  static constexpr int K_INTERLEAVE = 4;
  static constexpr int BLOCK_SIZE = 256;
  dim3 num_blocks(n / N_PER_BLOCK / K_INTERLEAVE);
  dim3 num_threads(BLOCK_SIZE);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      _in_feats.scalar_type(),
      "awq_gemv_forward_cuda",
      [&]
      {
        using f16_t = typename to_cpp_t<scalar_t>::type;
        auto in_feats = reinterpret_cast<f16_t *>(_in_feats.data_ptr());
        auto kernel = reinterpret_cast<uint32_t *>(_kernel.data_ptr());
        auto zeros = reinterpret_cast<f16_t *>(_zeros.data_ptr());
        auto scaling_factors = reinterpret_cast<f16_t *>(_scaling_factors.data_ptr());
        auto out_feats = reinterpret_cast<f16_t *>(_out_feats.data_ptr());

        if (group_size == 128)
        {
          switch (m)
          {
          case 1:
            gemv_kernel<f16_t, N_PER_BLOCK, 1, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
                in_feats, kernel, scaling_factors, zeros, out_feats, k, n);
            break;
          case 2:
            gemv_kernel<f16_t, N_PER_BLOCK, 2, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
                in_feats, kernel, scaling_factors, zeros, out_feats, k, n);
            break;
          case 3:
            gemv_kernel<f16_t, N_PER_BLOCK, 3, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
                in_feats, kernel, scaling_factors, zeros, out_feats, k, n);
            break;
          case 4:
            gemv_kernel<f16_t, N_PER_BLOCK, 4, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
                in_feats, kernel, scaling_factors, zeros, out_feats, k, n);
            break;
          case 5:
            gemv_kernel<f16_t, N_PER_BLOCK, 5, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
                in_feats, kernel, scaling_factors, zeros, out_feats, k, n);
            break;
          case 6:
            gemv_kernel<f16_t, N_PER_BLOCK, 6, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
                in_feats, kernel, scaling_factors, zeros, out_feats, k, n);
            break;
          case 7:
            gemv_kernel<f16_t, N_PER_BLOCK, 7, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
                in_feats, kernel, scaling_factors, zeros, out_feats, k, n);
            break;
          default:
            throw std::runtime_error("Unsupported batch size for gemv kernel.\n");
          }
        }
        else
        {
          throw std::runtime_error("Unsupported group size for gemv kernel.\n");
        }
      });

  return _out_feats;
}
