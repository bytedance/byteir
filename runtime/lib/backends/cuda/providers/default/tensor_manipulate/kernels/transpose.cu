//===- transpose.cu -------------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include <cuda_fp16.h>
#include <stdio.h>

namespace brt {
namespace cuda {
namespace kernel {
constexpr int32_t kMaxGridDim = 65535;
template <typename T>
__global__ void transpose_naive_2d_kernel(const T *input, T *output, int m,
                                          int n) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (iy < m && ix < n) {
    int in_idx = iy * n + ix;
    int out_idx = ix * m + iy;
    output[out_idx] = input[in_idx];
  }
}

template <typename T>
void transpose_naive_2d(const T *input, T *output, int m, int n, dim3 grid,
                        dim3 block, cudaStream_t stream) {
  transpose_naive_2d_kernel<T><<<grid, block, 0, stream>>>(input, output, m, n);
}

template <typename T, int32_t TileSizeX, int32_t TileSizeY, int32_t BlockSize>
__global__ void batch_transpose_kernel(const int32_t total_tile_num,
                                       const int32_t tile_num_in_dim0,
                                       const int32_t tile_num_in_dim1,
                                       const int32_t tile_per_sample,
                                       const int32_t row, const int32_t col,
                                       void *__restrict__ inp_ptr,
                                       void *__restrict__ out_ptr) {
  __shared__ T tile_in_shmem[TileSizeX][TileSizeY];
  for (int32_t i = blockIdx.x, step_tile = gridDim.x; i < total_tile_num;
       i += step_tile) {
    const int32_t batch_idx = i / tile_per_sample;
    const int32_t remainder = i - batch_idx * tile_per_sample;
    const int32_t dim0_idx = remainder / tile_num_in_dim1;
    const int32_t dim1_idx = remainder - dim0_idx * tile_num_in_dim1;

    T *inp_tile_gmem = reinterpret_cast<T *>(inp_ptr);
    T *out_tile_gmem = reinterpret_cast<T *>(out_ptr);
    inp_tile_gmem += batch_idx * row * col + dim0_idx * TileSizeX * col +
                     dim1_idx * TileSizeY;
    out_tile_gmem += batch_idx * row * col + dim1_idx * TileSizeY * row +
                     dim0_idx * TileSizeX;

    int32_t range_0 = dim0_idx < tile_num_in_dim0 - 1
                          ? TileSizeX
                          : row - dim0_idx * TileSizeX;
    int32_t range_1 = dim1_idx < tile_num_in_dim1 - 1
                          ? TileSizeY
                          : col - dim1_idx * TileSizeY;
    constexpr int32_t row_num_per_iter = BlockSize / TileSizeY;
    constexpr int32_t col_num_per_iter = BlockSize / TileSizeX;

    int32_t tile_row_idx = threadIdx.x / TileSizeY;
    int32_t tile_col_idx = threadIdx.x - tile_row_idx * TileSizeY;
    for (int32_t j = tile_row_idx; j < range_0; j += row_num_per_iter) {
      if (tile_col_idx < range_1) {
        tile_in_shmem[j][tile_col_idx ^ j] =
            inp_tile_gmem[j * col + tile_col_idx];
      }
    }
    __syncthreads();
    tile_row_idx = threadIdx.x / TileSizeX;
    tile_col_idx = threadIdx.x - tile_row_idx * TileSizeX;
    for (int32_t j = tile_row_idx; j < range_1; j += col_num_per_iter) {
      if (tile_col_idx < range_0) {
        out_tile_gmem[j * row + tile_col_idx] =
            tile_in_shmem[tile_col_idx][j ^ tile_col_idx];
      }
    }
    __syncthreads();
  }
}

template <typename T>
void batch_transpose(int32_t batch, int32_t row, int32_t col, const T *inp_ptr,
                     T *out_ptr, cudaStream_t stream) {
  constexpr int32_t kTileSize = 32;

  const int32_t tile_num_in_dim0 = (row - 1) / kTileSize + 1;
  const int32_t tile_num_in_dim1 = (col - 1) / kTileSize + 1;
  const int32_t tile_per_sample = tile_num_in_dim0 * tile_num_in_dim1;
  const int32_t total_tile_num = batch * tile_per_sample;
  dim3 grid(total_tile_num >= kMaxGridDim ? kMaxGridDim : total_tile_num);
  if (row < 8 || col < 8) {
    constexpr int32_t kBlockSize = 64;
    dim3 block(kBlockSize);
    batch_transpose_kernel<T, kTileSize, kTileSize, kBlockSize>
        <<<grid, block, 0, stream>>>(
            total_tile_num, tile_num_in_dim0, tile_num_in_dim1, tile_per_sample,
            row, col, reinterpret_cast<void *>(const_cast<T *>(inp_ptr)),
            reinterpret_cast<void *>(out_ptr));
  } else {
    constexpr int32_t kBlockSize = 256;
    dim3 block(kBlockSize);
    batch_transpose_kernel<T, kTileSize, kTileSize, kBlockSize>
        <<<grid, block, 0, stream>>>(
            total_tile_num, tile_num_in_dim0, tile_num_in_dim1, tile_per_sample,
            row, col, reinterpret_cast<void *>(const_cast<T *>(inp_ptr)),
            reinterpret_cast<void *>(out_ptr));
  }
}

// instantiate
template void transpose_naive_2d<float>(const float *, float *, int, int, dim3,
                                        dim3, cudaStream_t);
template void transpose_naive_2d<__half>(const __half *, __half *, int, int,
                                         dim3, dim3, cudaStream_t);
template void batch_transpose<float>(int32_t, int32_t, int32_t, const float *,
                                     float *, cudaStream_t);

template void batch_transpose<__half>(int32_t, int32_t, int32_t, const __half *,
                                      __half *, cudaStream_t);
} // namespace kernel
} // namespace cuda
} // namespace brt
