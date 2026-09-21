# 错误模式追踪

## 01 — Multi-Head Attention(2026-09-21,开始前自报的历史错误)

- **[shape/内存布局]** 把 (B, T, d_model) 直接 `view` 成 (B, H, T, hd),导致内存错乱:shape 对但数值全错,测试才能发现。
- **[API 记忆]** `nn.Linear` 误传 dim 类参数,把它当成带 dim/axis 的算子。
- **[命名]** 局部变量名覆盖了 module 属性,后面再用时拿到的已经不是 module。
- **[内存布局]** `.contiguous()` 放错位置。
- **[范围蔓延]** 往题目里加没要求的 LayerNorm / dropout,面试里等于多写多错。
