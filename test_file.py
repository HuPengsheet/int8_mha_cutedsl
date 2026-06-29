import torch
import math
from int8_sdpa import reference_int8_sdpa

def quantize(tensor, scale=None):
    """对称量化到 int8，自动计算 scale 或使用给定值"""
    if scale is None:
        scale = tensor.abs().max() / 127.0
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale

def main():
    # 使用短序列，使 softmax 平均概率增大
    B, H, S, D = 1, 8, 1024, 64          # 序列长度从 1024 改为 16
    Dv = 64

    torch.manual_seed(42)
    # 增大 q/k 的方差，使注意力分数差异更明显
    amp = 5.0                           # 放大系数，使 softmax 更陡峭
    q_fp = torch.randn(B, H, S, D) * amp
    k_fp = torch.randn(B, H, S, D) * amp
    v_fp = torch.randn(B, H, S, Dv)

    # 量化到 int8
    q_int8, q_scale = quantize(q_fp)
    k_int8, k_scale = quantize(k_fp)
    v_int8, v_scale = quantize(v_fp)

    # 输出量化缩放因子（任意设定）
    out_scale = 0.1
    inv_out_scale = 1.0 / out_scale

    # 调用参考实现
    out_int8 = reference_int8_sdpa(
        q=q_int8,
        k=k_int8,
        v=v_int8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        inv_scale_out=inv_out_scale,
        dim=float(D)
    )


    print(out_int8)



if __name__ == "__main__":
    main()