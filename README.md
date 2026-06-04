# Q3Q

> Qwen3.5 MoE text-only inference, written from scratch with Triton-fused kernels.

个人学习项目：从零实现 Qwen3.5 的纯文本推理，逐步用 Triton kernel 替换原生 PyTorch 实现，目标是把整个 forward 路径融合成尽可能大的 kernel

博客笔记：https://www.xiwon.blog/p/qwen3.5-%E8%A7%82%E6%B5%8B%E8%AE%B0%E5%BD%95/

## 项目结构

```
q3q/
├── eager_main.py        # baseline，PyTorch
├── eager_split.py
├── eager_opt.py         # 添加了 full-attention 和 GDN 的 cache 管理
├── kernelized.py        # in progress: 逐步用 Triton kernel 替换算子
├── kernels/             # Triton kernels
│   ├── rms_norm.py
│   └── rms_norm_gated.py
├── tests/parity/
│   └── test_parity.py
├── bench/
│   └── bench.py
└── q35_35b_a3b/         # weights (gitignored)
```

## Roadmap

| Phase | 状态 | 内容 |
|---|---|---|
| 0 | done | prefill / decode 路径拆分 + parity & bench harness |
| 1 | done | full-attention KV cache、GDN recurrent state + conv1d 滑窗 cache |
| 2 | in progress | 单算子 Triton kernel 化（RMSNorm、RMSNormGated、partial RoPE、l2norm…） |
| 3 | next | 垂直融合：FlashAttention prefill/decode、fused MoE、chunkwise GDN prefill |
| 4 | later | 水平融合：residual+norm+qkv prologue、attn+oproj epilogue、post_norm+router 等 |
| 5 | optional | CUDA Graph for decode；persistent megakernel 探索 |