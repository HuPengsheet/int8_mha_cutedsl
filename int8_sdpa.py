import torch
import math



import torch
import math
from typing import Optional

def reference_int8_sdpa(
    q: torch.Tensor,          # [batch, heads, seq_q, dim]  int8
    k: torch.Tensor,          # [batch, heads, seq_k, dim]  int8
    v: torch.Tensor,          # [batch, heads, seq_k, dim_v] int8
    q_scale: float,           # 反量化 q 的缩放因子（即 q = q_int8 * q_scale）
    k_scale: float,           # 反量化 k 的缩放因子
    v_scale: float,           # 反量化 v 的缩放因子
    inv_scale_out: float,     # 输出反量化缩放因子的倒数（即 1 / scale_out）
    dim: float,               # 注意力头维度（用于缩放 sqrt(dim)）
) -> torch.Tensor:            # [batch, heads, seq_q, dim_v] int8
    """
    模拟硬件中 int8 量化的 SDPA（Scaled Dot-Product Attention）计算流程。
    整个计算过程严格按照 CUDA 内核中的顺序执行，用于验证实际硬件实现的正确性。

    步骤说明：
      1. 将 q, k, v 提升为 int32 以避免溢出（模拟 MMA 累积）
      2. 计算 q @ k^T 得到 int32 分数矩阵
      3. 反量化分数并除以 sqrt(dim) 得到浮点分数
      4. 对浮点分数做 softmax（在浮点域）
      5. 将 softmax 输出量化到 int8：乘以 127，四舍五入，钳制到 [-128, 127]
      6. 计算 attn_int8 @ v_int32 得到 int32 累加输出
      7. 将累加输出反量化到浮点：乘以 (1/127) * v_scale / scale_out
      8. 钳制到 [-128, 127] 并转换为 int8

    注意：
      - 参数 inv_scale_out 应为 1 / scale_out，其中 scale_out 是输出张量的量化缩放因子。
        在 CUDA 代码中，最终缩放为 softmax_quant_scale * v_scale / scale_out，
        即 (1/127) * v_scale * (1/scale_out)。
      - 舍入方式使用 torch.round()，对应 CUDA 的 __float2int_rn（四舍五入到最近偶数）。
      - 钳制范围与硬件一致：[-128, 127]。
    """
    # 1. 提升到 int32（模拟 MMA 累加器）
    q_int32 = q.to(torch.int32)
    k_int32 = k.to(torch.int32)
    v_int32 = v.to(torch.int32)

    # 2. 第一次矩阵乘法：q * k^T，得到 int32 分数
    scores_int32 = torch.matmul(q_int32, k_int32.transpose(-2, -1))

    # 3. 反量化到浮点并除以 sqrt(dim)
    scores_f = scores_int32.to(torch.float32) * (q_scale * k_scale) / math.sqrt(dim)

    # 4. 浮点 softmax
    attn_f = torch.softmax(scores_f, dim=-1)

    # 5. 将 attention 权重量化到 int8（乘以 127，四舍五入，钳制）
    temp_scale = 1.0 / 127.0
    attn_int8 = (attn_f / temp_scale).round().clamp(-128, 127).to(torch.int8)   # 等价于 round(attn_f * 127)

    # 6. 第二次矩阵乘法：attn_int8 * v_int32，得到 int32 输出
    out_int32 = torch.matmul(attn_int8.to(torch.int32), v_int32)

    # 7. 反量化输出到浮点：乘以 (1/127) * v_scale * inv_scale_out
    out_f = out_int32.to(torch.float32) * temp_scale * v_scale * inv_scale_out

    # 8. 钳制并转换为 int8
    out_int8 = out_f.clamp(-128, 127).to(torch.int8)

    return out_int8




