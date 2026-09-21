# 01 — Multi-Head Attention

**限时:25 分钟** · 解答文件:`solutions/01_mha.py` · 测试:`tests/test_01.py`

## 任务

用 PyTorch 实现一个 `MultiHeadAttention` 模块(self-attention,标准 scaled dot-product attention,
语义与 `torch.nn.MultiheadAttention` 一致,缩放因子 1/√d_head)。

## 接口

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int): ...

    def forward(
        self,
        x: torch.Tensor,                           # (B, T, d_model)
        causal: bool = False,
        padding_mask: torch.Tensor | None = None,  # (B, T) bool,True = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
        return output, attn_weights                # (B, T, d_model), (B, num_heads, T, T)
```

- 四个投影层为 `nn.Linear(d_model, d_model)`(带 bias),属性名固定为
  `q_proj`、`k_proj`、`v_proj`、`out_proj`。测试靠这四个名字把权重拷进
  `torch.nn.MultiheadAttention` 做数值对比,名字不对测试跑不了。
- `d_model` 不能被 `num_heads` 整除时,构造函数必须抛异常(`AssertionError` 或 `ValueError` 均可)。

## 语义

- `attn_weights[b, h, i, j]`:第 b 个样本、第 h 个头里,query 位置 i 对 key 位置 j 的 softmax 后权重。
- `causal=True`:位置 i 只能 attend 到 j ≤ i。
- `padding_mask[b, j] == True`:key 位置 j 不可被任何 query attend,对应权重为 0。
  padding 位置作为 **query** 的那些行照常计算,不做特殊处理(和 `key_padding_mask` 语义一致)。
- 两个 mask 可以同时传,效果取交集:只有两个都允许的 (i, j) 才可 attend。
- 无 mask 时,`output` 和 `attn_weights` 都要与拷贝了权重的
  `torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True)` 数值接近(atol 1e-5)。
  有 mask 且没有整行被 mask 的情况下也要一致。

## 数值要求

- softmax 数值稳定:输入量级很大(例如 `x * 1e4`)时不能出 NaN/Inf。
- **全被 mask 的行**(例如左 padding + causal 下前几个 query 位置,或整条序列全是 padding)
  不能出 NaN/Inf。这些行的权重取值不做规定(全 0 或均匀都可以),
  但 `output` 和 `attn_weights` 必须全是有限值。
- 其余行 attn 权重非负、每行和为 1。

## 不要做

- 不要 dropout,不要 KV cache,不要 LayerNorm,不要残差连接。只做题目要求的事。
