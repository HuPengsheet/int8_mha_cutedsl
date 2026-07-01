# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import time
from typing import Type
import cutlass.utils as utils

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
import torch
import cutlass.torch as cutlass_torch

#256 128

@cute.kernel
def run_kernel(a_tensor , b_tensor , tiled_copy_a ,tiled_copy_b,s_layout):
    # 在 shared memory 中分配一块和 s_layout 同形状的 tensor
    smem = cutlass.utils.SmemAllocator()
    c_smem = smem.allocate_tensor(a_tensor.element_type, s_layout, byte_alignment=16)

    # 计算当前 block 负责的 tile 坐标
    # a_tensor: (2048, 4096), tile: (256, 128) -> 8 x 32 = 256 个 tile
    bx, _, _ = cute.arch.block_idx()
    tile_n_count = 32  # 4096 // 128
    tile_m = bx // tile_n_count
    tile_n = bx % tile_n_count

    # 取出本 block 在 gmem 中的 tile 视图
    tile_shape = cute.shape(s_layout)
    gA_tile = cute.local_tile(a_tensor, tile_shape, (tile_m, tile_n))
    gB_tile = cute.local_tile(b_tensor, tile_shape, (tile_m, tile_n))

    tid, _, _ = cute.arch.thread_idx()

    # 1) G2S: a_tensor(gmem) -> c_smem
    thr_g2s = tiled_copy_a.get_slice(tid)
    gA_part = thr_g2s.partition_S(gA_tile)
    sA_part = thr_g2s.partition_D(c_smem)
    cute.copy(tiled_copy_a, gA_part, sA_part)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # 2) S2G: c_smem -> b_tensor(gmem)
    thr_s2g = tiled_copy_b.get_slice(tid)
    sB_part = thr_s2g.partition_S(c_smem)
    gB_part = thr_s2g.partition_D(gB_tile)
    cute.copy(tiled_copy_b, sB_part, gB_part)
    cute.arch.barrier()


@cute.jit
def run(a_tensor, b_tensor):
  num_vectorized = 8
  # G2S: 用 cp.async 异步拷贝
  atom_async_copy_A = cute.make_copy_atom(
      cute.nvgpu.cpasync.CopyG2SOp(),
      a_tensor.element_type,
      num_bits_per_copy=a_tensor.element_type.width * num_vectorized,
  )
  # S2G: 用通用的同步拷贝 (register 中转)
  atom_s2g = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      a_tensor.element_type,
      num_bits_per_copy=a_tensor.element_type.width * num_vectorized,
  )

  thread_layout = cute.make_layout((16,16),stride=(16,1))
  value_layout =  cute.make_layout((16,8),stride=(8,1))
  tiled_copy_a = cute.make_tiled_copy_tv(atom_async_copy_A, thread_layout, value_layout)
  tiled_copy_b = cute.make_tiled_copy_tv(atom_s2g, thread_layout, value_layout)

  s_layout = cute.make_layout((256,128),stride = (128,1))
  run_kernel(a_tensor, b_tensor, tiled_copy_a, tiled_copy_b, s_layout).launch(grid=(256,1,1),block=[256,1,1])

a = torch.rand(2048,4096).cuda().half()
b = torch.zeros(2048,4096).cuda().half()
a_tensor = from_dlpack(a, assumed_align=16)
b_tensor = from_dlpack(b, assumed_align=16)

run(a_tensor, b_tensor)

# 比较 a 和 b 是否相等 (b 应该等于 a, 因为经过 gmem->smem->gmem 的拷贝)
torch.cuda.synchronize()
print("equal:", torch.equal(a, b))
print("max_diff:", (a - b).abs().max().item())
