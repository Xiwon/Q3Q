"""
Q3Q :: Qwen3.5 MoE Text-Only Inference (Eager, prefill/decode split, optimized)

* `QMoeEngine` exposes explicit prefill and decode_step entry points

The state_dict is named to match the checkpoint exactly.
The `mtp.*` layer and `model.visual.*` shards are skipped.
"""

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

class CacheFullAttention:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys # (B, num_heads, T, head_dim)
        self.values = values
    
    def update(self, key_t: torch.Tensor, value_t: torch.Tensor):
        # assume key_t and value_t as (B, num_heads, 1, head_dim)
        self.keys = torch.cat([self.keys, key_t], dim=-2)
        self.values = torch.cat([self.values, value_t], dim=-2)
        return self.keys, self.values

class CacheGatedDeltaNet:
    def __init__(self, state: torch.Tensor, mixed_cache: torch.Tensor):
        self.state = state              # (B, num_v_heads, k_head_dim, v_head_dim)
        self.mixed_cache = mixed_cache  # (B, qkv_dim, conv_kernel_size)

    def update(
        self,
        mixed_t: torch.Tensor | None = None, # (B, qkv_dim, 1)
        state_t: torch.Tensor | None = None,
    ):
        if mixed_t != None:
            self.mixed_cache = torch.cat([self.mixed_cache[:, :, 1:], mixed_t], dim=-1)
            return self.mixed_cache, self.state
        if state_t != None:
            self.state = state_t

class Cache:
    def __init__(self):
        self.layer_cache = []

    def append(self, layer: CacheFullAttention | CacheGatedDeltaNet): # for prefill usage
        self.layer_cache.append(layer)

    def update(self, layer_idx: int, **kwargs): # for decode usage
        return self.layer_cache[layer_idx].update(**kwargs)
    
    def size(self):
        return len(self.layer_cache)

class QMoeConfig:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        text = cfg.get("text_config", cfg)

        self.vocab_size = text["vocab_size"]
        self.hidden_size = text["hidden_size"]
        self.num_hidden_layers = text["num_hidden_layers"]
        self.num_attention_heads = text["num_attention_heads"]
        self.num_key_value_heads = text["num_key_value_heads"]
        self.head_dim = text["head_dim"]
        self.max_position_embeddings = text["max_position_embeddings"]
        self.rms_norm_eps = text["rms_norm_eps"]

        rope = text.get("rope_parameters", {})
        self.rope_theta = rope.get("rope_theta", 10000.0)
        self.partial_rotary_factor = rope.get("partial_rotary_factor", 1.0)

        self.linear_conv_kernel_dim = text["linear_conv_kernel_dim"]
        self.linear_key_head_dim = text["linear_key_head_dim"]
        self.linear_value_head_dim = text["linear_value_head_dim"]
        self.linear_num_key_heads = text["linear_num_key_heads"]
        self.linear_num_value_heads = text["linear_num_value_heads"]

        self.num_experts = text["num_experts"]
        self.num_experts_per_tok = text["num_experts_per_tok"]
        self.moe_intermediate_size = text["moe_intermediate_size"]
        self.shared_expert_intermediate_size = text["shared_expert_intermediate_size"]

        self.layer_types = text["layer_types"]
        self.tie_word_embeddings = cfg.get("tie_word_embeddings", False)

class RMSNorm(nn.Module):
    """output = ((x_fp32 / rms) * (1 + w_fp32)).to(orig)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        x_f = x_f * torch.rsqrt(var + self.eps)
        out = x_f * (1.0 + self.weight.float())
        return out.to(orig_dtype)

class RMSNormGated(nn.Module):
    """Per-head RMSNorm with silu output gate"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        x_f = x_f * torch.rsqrt(var + self.eps)
        out = (self.weight * x_f).to(orig_dtype)
        out = out * F.silu(gate.float()).to(orig_dtype)
        return out

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_out = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_out = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_out, k_out

def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    """Gated delta-rule linear attention

    * conv1d -> silu over `mixed_qkv`, depthwise, kernel 4, no bias
    * split into Q, K, V; reshape to per-head; if num_v_heads > num_k_heads,
      repeat-interleave Q and K along the head axis.
    * beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
    * q,k go through l2 norm before scaling

    checkpoint layout (per layer, prefix `linear_attn.`):
        in_proj_qkv.weight  (qk*nk + qk*nk + vk*nv, hidden) packed Q|K|V
        in_proj_z.weight    (nv*vk, hidden)         output gate (per-head, silu)
        in_proj_b.weight    (nv,    hidden)         beta logits
        in_proj_a.weight    (nv,    hidden)         decay logits
        conv1d.weight       (qkv_dim, 1, k)         depthwise causal conv (no bias)
        A_log               (nv,)   fp32
        dt_bias             (nv,)
        out_proj.weight     (hidden, nv*vk)
        norm.weight         (vk,)   fp32   per-head RMSNorm with (1 + w)
    """

    def __init__(self, config: QMoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.k_head_dim = config.linear_key_head_dim
        self.v_head_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.qk_dim = self.num_k_heads * self.k_head_dim
        self.v_dim = self.num_v_heads * self.v_head_dim
        self.qkv_dim = self.qk_dim + self.qk_dim + self.v_dim

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.v_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.qkv_dim,
            out_channels=self.qkv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.qkv_dim,
            bias=False,
            padding=0,
        )

        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))

        self.out_proj = nn.Linear(self.v_dim, self.hidden_size, bias=False)
        self.norm = RMSNormGated(self.v_head_dim, config.rms_norm_eps)

    def prefill(self, hidden_states: torch.Tensor, cache: Cache) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype

        # in `mixed` tensor, every token's vector contain 3 parts representing qkv
        mixed = self.in_proj_qkv(hidden_states)              # -> (B, T, qkv_dim)
        mixed = mixed.transpose(1, 2)                        # -> (B, qkv_dim, T)
        mixed = F.pad(mixed, (self.conv_kernel_size - 1, 0)) # pad conv_kernel_size-1 zeros left, 0 zeros right

        conv_cache = (mixed.clone())[:, :, -self.conv_kernel_size:]
        # take last conv_kernel_size vectors to save

        mixed = self.conv1d(mixed)                           # every dimension as a channel
        mixed = F.silu(mixed)
        mixed = mixed.transpose(1, 2)                        # -> (B, T, qkv_dim)

        q, k, v = mixed.split([self.qk_dim, self.qk_dim, self.v_dim], dim=-1)
        q = q.view(B, T, self.num_k_heads, self.k_head_dim)
        k = k.view(B, T, self.num_k_heads, self.k_head_dim)
        v = v.view(B, T, self.num_v_heads, self.v_head_dim)

        if self.num_v_heads > self.num_k_heads:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
            # !note: q, k -> (B, T, num_v_heads, k_head_dim) from now on

        # z is the per-head output gate -> (B, T, num_v_heads, v_head_dim).
        z = self.in_proj_z(hidden_states).reshape(B, T, self.num_v_heads, self.v_head_dim)

        b = self.in_proj_b(hidden_states)   # -> (B, T, num_v_heads)
        a = self.in_proj_a(hidden_states)   # -> (B, T, num_v_heads)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        q = _l2norm(q.float(), eps=1e-6)
        k = _l2norm(k.float(), eps=1e-6)
        v = v.float()
        beta = beta.float()
        scale = 1.0 / math.sqrt(self.k_head_dim)
        q = q * scale

        # recurrent delta-rule (eager)
        # state -> (B, num_v_heads, k_head_dim, v_head_dim).
        state = torch.zeros(
            B, self.num_v_heads, self.k_head_dim, self.v_head_dim,
            dtype=torch.float32, device=hidden_states.device,
        )
        out = torch.empty(
            B, T, self.num_v_heads, self.v_head_dim,
            dtype=torch.float32, device=hidden_states.device,
        )

        for t in range(T):
            q_t = q[:, t] # (B, num_v_heads, k_head_dim) !note: qk was expanded before
            k_t = k[:, t] # (B, num_v_heads, k_head_dim)
            v_t = v[:, t] # (B, num_v_heads, v_head_dim)
            
            # last_recurrent_state *= exp(g_t)
            g_t = g[:, t].exp()[:, :, None, None] # (B, num_v_heads, 1, 1)
            state = state * g_t

            # delta = (v - state^T k) * beta. !note: here we use *key* for read
            beta_t = beta[:, t].unsqueeze(-1) # (B, num_v_heads, 1)
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2) # (B, nv, v_head_dim)
            delta = (v_t - kv_mem) * beta_t                  # (B, nv, v_head_dim)
            # state += k * delta
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            # output = sum_d state[..., d, :] * q[..., d]
            out[:, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        
        cache.append(CacheGatedDeltaNet(state, conv_cache))
        # build up cache

        out = out.to(orig_dtype) # (B, T, num_v_heads, v_head_dim)
        # norm over v_head_dim, then silu-gate by z
        out = self.norm(out, z.to(orig_dtype))
        out = out.reshape(B, T, self.v_dim)
        return self.out_proj(out)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int, cache: Cache) -> torch.Tensor:
        h = hidden_states
        B, T, _ = h.shape
        orig_dtype = h.dtype

        # in `mixed` tensor, every token's vector contain 3 parts representing qkv
        mixed = self.in_proj_qkv(h)     # -> (B, T, qkv_dim)
        mixed = mixed.transpose(1, 2)   # -> (B, qkv_dim, T)
        
        mixed, state = cache.update(layer_idx, mixed_t=mixed)
        # update current vector & get current conv1d values and last state cache
        mixed = self.conv1d(mixed)
        
        mixed = F.silu(mixed)
        mixed = mixed.transpose(1, 2)                        # -> (B, T, qkv_dim)
        # print(f"[GatedDeltaNet.forward]> mixed.shape = {mixed.shape}")
        # assert T == 1

        q, k, v = mixed.split([self.qk_dim, self.qk_dim, self.v_dim], dim=-1)
        q = q.view(B, T, self.num_k_heads, self.k_head_dim)
        k = k.view(B, T, self.num_k_heads, self.k_head_dim)
        v = v.view(B, T, self.num_v_heads, self.v_head_dim)
        # print(f"[GatedDeltaNet.forward]> q.shape = {q.shape}")

        if self.num_v_heads > self.num_k_heads:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
            # !note: q, k -> (B, T, num_v_heads, k_head_dim) from now on

        # z is the per-head output gate -> (B, T, num_v_heads, v_head_dim).
        z = self.in_proj_z(h).reshape(B, T, self.num_v_heads, self.v_head_dim)

        b = self.in_proj_b(h)   # -> (B, T, num_v_heads)
        a = self.in_proj_a(h)   # -> (B, T, num_v_heads)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        q = _l2norm(q.float(), eps=1e-6)
        k = _l2norm(k.float(), eps=1e-6)
        v = v.float()
        beta = beta.float()
        scale = 1.0 / math.sqrt(self.k_head_dim)
        q = q * scale

        q_t = q.squeeze(1) # (B, num_v_heads, k_head_dim) !note: qk was expanded before
        k_t = k.squeeze(1) # (B, num_v_heads, k_head_dim)
        v_t = v.squeeze(1) # (B, num_v_heads, v_head_dim)
        
        # last_recurrent_state *= exp(g_t)
        g_t = g.squeeze(1).exp()[:, :, None, None] # (B, num_v_heads, 1, 1)
        state = state * g_t

        # delta = (v - state^T k) * beta. !note: here we use *key* for read
        beta_t = beta.squeeze(1).unsqueeze(-1) # (B, num_v_heads, 1)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2) # (B, nv, v_head_dim)
        delta = (v_t - kv_mem) * beta_t                  # (B, nv, v_head_dim)
        # state += k * delta
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        cache.update(layer_idx, state_t=state)

        out = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        out = out.to(orig_dtype) # (B, T, num_v_heads, v_head_dim)
        # norm over v_head_dim, then silu-gate by z
        out = self.norm(out, z.to(orig_dtype))
        out = out.reshape(B, T, self.v_dim)
        out = self.out_proj(out)
        return out

class FullAttention(nn.Module):
    """Standard softmax attention

    checkpoint layout (per layer, prefix `self_attn.`):
        q_proj.weight    (2 * nh * hd, hidden)   packed Q | output-gate
        k_proj.weight    (nkv * hd,    hidden)
        v_proj.weight    (nkv * hd,    hidden)
        o_proj.weight    (hidden,      nh * hd)
        q_norm.weight    (hd,)                    per-head RMSNorm
        k_norm.weight    (hd,)
    """

    def __init__(self, config: QMoeConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, 2 * self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def prefill(
        self,
        hidden_states: torch.Tensor,
        cache: Cache,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        # q_proj packs Q and gate INSIDE each head: shape (B, T, num_heads, 2*head_dim).
        qg = self.q_proj(hidden_states).view(B, T, self.num_heads, 2 * self.head_dim)
        q, gate = qg.chunk(2, dim=-1) # each (B, T, num_heads, head_dim)
        gate = gate.reshape(B, T, self.num_heads * self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2) # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        cache.append(CacheFullAttention(k.clone(), v.clone()))
        # build up kv cache

        # assume num_q_heads is a multiple of num_kv_heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        cache: Cache,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = hidden_states
        B, T, _ = h.shape

        # q_proj packs Q and gate INSIDE each head: shape (B, T, num_heads, 2*head_dim).
        qg = self.q_proj(h).view(B, T, self.num_heads, 2 * self.head_dim)
        q, gate = qg.chunk(2, dim=-1) # each (B, T, num_heads, head_dim)
        gate = gate.reshape(B, T, self.num_heads * self.head_dim)
        k = self.k_proj(h).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2) # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2) # for decode step, always assume T == 1
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k, v = cache.update(layer_idx, key_t=k, value_t=v)
        # use kv cache

        # assume num_q_heads is a multiple of num_kv_heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # q     (B, num_heads, 1, head_dim)
        # k, v  (B, num_heads, T, head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (B, num_heads, 1, T)

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        # (B, num_heads, 1, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        out = self.o_proj(out)
        return out

class QMoeMLP(nn.Module):
    """MoE block matching the checkpoint layout.

    checkpoint layout (per layer, prefix `mlp.`):
        gate.weight                       (num_experts, hidden)
        experts.gate_up_proj              (num_experts, 2*moe_inter, hidden)
        experts.down_proj                 (num_experts, hidden, moe_inter)
        shared_expert.gate_proj.weight    (shared_inter, hidden)
        shared_expert.up_proj.weight      (shared_inter, hidden)
        shared_expert.down_proj.weight    (hidden, shared_inter)
        shared_expert_gate.weight         (1, hidden)
    """

    def __init__(self, config: QMoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_inter = config.moe_intermediate_size

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        class _Experts(nn.Module):
            def __init__(self, num_experts: int, hidden: int, inter: int):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * inter, hidden))
                self.down_proj = nn.Parameter(torch.empty(num_experts, hidden, inter))

        self.experts = _Experts(self.num_experts, self.hidden_size, self.moe_inter)

        class _SharedExpert(nn.Module):
            def __init__(self, hidden: int, inter: int):
                super().__init__()
                self.gate_proj = nn.Linear(hidden, inter, bias=False)
                self.up_proj = nn.Linear(hidden, inter, bias=False)
                self.down_proj = nn.Linear(inter, hidden, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        self.shared_expert = _SharedExpert(self.hidden_size, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        flat = hidden_states.view(-1, H) # -> (B * T, hidden_size)

        router_logits = self.gate(flat) # -> (B * T, num_experts)
        routing_weights = F.softmax(router_logits.float(), dim=-1) # -> (B * T, num_experts)
        topk_w, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1) # -> (B * T, top_k)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True) # weights norm
        topk_w = topk_w.to(hidden_states.dtype)

        out = torch.zeros_like(flat) # -> (B * T, hidden_size)

        expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).permute(2, 1, 0)
        # .one_hot -> (B * T, top_k, num_experts)
        # .permute -> (num_experts, top_k, B * T)
        active = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).squeeze(-1)
        # .greater -> (num_experts)
        # .nonzero -> (<non_zero_nums>, 1)
        # .squeeze -> (<non_zero_nums>)
        for e_idx in active.tolist():
            slot, tok = torch.where(expert_mask[e_idx])
            # expert_mask[e_idx] -> (top_k, B * T)
            # .where to return non-zero positions
            # slot -> (<e_idx_select_count>) : indices of `top_k` dimension
            # tok  -> (<e_idx_select_count>) : indices of `B * T` dimension
            x = flat[tok]
            gu = F.linear(x, self.experts.gate_up_proj[e_idx])
            g, u = gu.chunk(2, dim=-1)
            y = F.linear(F.silu(g) * u, self.experts.down_proj[e_idx])
            # take vector from `flat` and perform SwiGLU
            y = y * topk_w[tok, slot, None]
            out.index_add_(0, tok, y)

        shared = self.shared_expert(flat)
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        out = out + shared_gate * shared
        return out.view(B, T, H)

class DecoderLayer(nn.Module):
    def __init__(self, config: QMoeConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = FullAttention(config)
        self.mlp = QMoeMLP(config)

    def prefill(
        self,
        hidden_states: torch.Tensor,
        cache: Cache,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # attention part
        residual = hidden_states # -> (B, T, hidden_size)
        x = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            x = self.linear_attn.prefill(x, cache)
        else:
            x = self.self_attn.prefill(x, cache, cos, sin, attention_mask)
        hidden_states = residual + x

        # MoE part
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        return residual + x

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        cache: Cache,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # attention part
        residual = hidden_states # -> (B, T, hidden_size)
        x = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, layer_idx, cache)
        else:
            x = self.self_attn(x, layer_idx, cache, cos, sin, attention_mask)
        hidden_states = residual + x

        # print(f"[DecoderLayer.forward]> hidden_states.shape = {hidden_states.shape}")

        # MoE part
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        return residual + x

class RotaryEmbedding(nn.Module): # partial rotary
    def __init__(self, head_dim: int, base: float, partial_rotary_factor: float):
        super().__init__()
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        ) # inv_freq -> (rotary_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv = self.inv_freq[None, :, None].float()  # inv -> (1, rotary_dim, 1)
        pos = position_ids[:, None, :].float()      # pos -> (B, 1, T)
        freqs = (inv @ pos).transpose(1, 2)         # freqs -> (B, T, rotaty_dim)
        emb = torch.cat((freqs, freqs), dim=-1)     # emb -> (B, T, rotary_dim * 2)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

class QMoeModel(nn.Module):
    def __init__(self, config: QMoeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.partial_rotary_factor
        )

    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, Cache]:
        B, T = input_ids.shape
        h = self.embed_tokens(input_ids) # h -> (B, T, hidden_size)
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        # position_ids -> (B, T): 0, 1, ... T-1  repeat for B times
        cos, sin = self.rotary_emb(h, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1) # cos, sin -> (B, 1, T, rotary_dim * 2)

        attention_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1
        ) # (i, j) set to -inf if j > i

        cache = Cache()
        for layer in self.layers:
            h = layer.prefill(h, cache, cos, sin, attention_mask)
        
        return self.norm(h), cache

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Cache,
        token_pos: int,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.embed_tokens(input_ids) # h -> (B, T, hidden_size)
        position_ids = torch.tensor([token_pos], device=input_ids.device).unsqueeze(0).expand(B, -1)
        # position_ids -> (B, T): 0, 1, ... T-1  repeat for B times
        cos, sin = self.rotary_emb(h, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1) # cos, sin -> (B, 1, T, rotary_dim * 2)

        attention_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1
        ) # (i, j) set to -inf if j > i

        for i, layer in enumerate(self.layers):
            h = layer(h, i, cache, cos, sin, attention_mask)
        return self.norm(h)

class QMoeEngine(nn.Module):
    """Engine with explicit prefill / decode_step entry points."""

    def __init__(self, config: QMoeConfig):
        super().__init__()
        self.config = config
        self.model = QMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        """Process the full prompt. Returns logits"""

        # h = self.model(input_ids, cache)
        h, cache = self.model.prefill(input_ids)
        # print(f"[QMoeEngine.prefill]> cache.size() = {cache.size()}")
        # print(f"[QMoeEngine.prefill]> h.shape = {h.shape}")
        logits = self.lm_head(h)
        return logits, cache

    @torch.no_grad()
    def decode_step(self, next_token: torch.Tensor, cache: Cache, token_pos: int):
        """Run a single decoding step. Returns logits_for_new_token"""
        
        h = self.model(next_token, cache, token_pos)
        logits = self.lm_head(h)
        return logits

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        next_logits = logits.float() / max(temperature, 1e-6)
        sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = remove.scatter(1, sorted_idx, remove)
        next_logits = next_logits.masked_fill(mask, float("-inf"))
        probs = F.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_ids: Optional[list] = None,
    ) -> torch.Tensor:
        eos_set = set(eos_token_ids or [])
        result = input_ids
        logits, cache = self.prefill(input_ids)
        prefill_token_num = input_ids.shape[1]
        for i in range(max_new_tokens):
            next_tok = self._sample(logits[:, -1, :], temperature, top_p)
            result = torch.cat([result, next_tok], dim=-1)
            if eos_set and next_tok.item() in eos_set:
                break
            logits = self.decode_step(next_tok, cache, prefill_token_num + i)
        return result

CHECKPOINT_PREFIX = "model.language_model."
SKIP_PREFIXES = ("mtp.", "model.visual.")

def _map_state_key(checkpoint_key: str) -> Optional[str]:
    """Map a checkpoint key to a state_dict key on QMoeEngine.
    Returns None for checkpoint keys that belong to discarded sub-modules
    """
    if any(checkpoint_key.startswith(p) for p in SKIP_PREFIXES):
        return None
    if checkpoint_key == "lm_head.weight":
        return "lm_head.weight"
    if checkpoint_key.startswith(CHECKPOINT_PREFIX):
        rest = checkpoint_key[len(CHECKPOINT_PREFIX):]
        return "model." + rest
    return None

def load_safetensors_weights(model: nn.Module, model_path: str, device: str = "cuda") -> None:
    """Stream weights one tensor at a time straight onto `device`.
    The model is expected to live on `device` already (typically materialised via `to_empty`).
    """
    
    model_path = Path(model_path)
    shards = sorted(model_path.glob("*.safetensors"))

    state = dict(model.state_dict())
    target_keys = set(state.keys())
    loaded = set()
    skipped = set()

    for shard_idx, shard_path in enumerate(shards):
        print(f"[{shard_idx + 1}/{len(shards)}] {shard_path.name}")
        with safe_open(shard_path, framework="pt", device=device) as f:
            for ckpt_key in f.keys():
                target = _map_state_key(ckpt_key)
                if target is None:
                    skipped.add(ckpt_key)
                    continue
                if target not in target_keys:
                    print(f"  ! checkpoint key has no model home: {ckpt_key} -> {target}")
                    continue

                tensor = f.get_tensor(ckpt_key)
                dst = state[target]
                if tensor.shape != dst.shape:
                    raise RuntimeError(
                        f"shape mismatch for {target}: ckpt {tuple(tensor.shape)} vs model {tuple(dst.shape)}"
                    )
                if tensor.dtype != dst.dtype:
                    tensor = tensor.to(dst.dtype)
                
                dst.copy_(tensor)
                loaded.add(target)
                del tensor

    if model.config.tie_word_embeddings and "lm_head.weight" not in loaded:
        model.lm_head.weight.data.copy_(model.model.embed_tokens.weight.data)
        loaded.add("lm_head.weight")

    missing = sorted(target_keys - loaded)
    if missing:
        print(f"WARNING: {len(missing)} model parameters were not loaded.")
        for k in missing[:20]:
            print(f"  - {k}")
    
    print(f"= skipped keys count: {len(skipped)}")
    print(f"= loaded {len(loaded)} / {len(target_keys)} parameters.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Q3Q :: Qwen3.5 MoE eager inference (prefill/decode split)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="skip the chat template and feed --prompt as-is. Useful only for base models or debugging",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="optional system message prepended to the chat template",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="enable the model's <think> reasoning block",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not args.device.startswith("cuda"):
        raise RuntimeError("only support cuda device")

    torch.set_default_dtype(torch.bfloat16)
    config_path = Path(args.model_path) / "config.json"
    config = QMoeConfig(str(config_path))

    print(f"= building model on meta ...")
    with torch.device("meta"):
        model = QMoeEngine(config)

    print(f"= allocating empty parameters on {args.device} ...")
    torch.set_default_dtype(torch.float32)
    model.to_empty(device=args.device)

    print(f"= streaming weights into {args.device} ...")
    load_safetensors_weights(model, args.model_path, device=args.device)
    model.eval()

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"= GPU memory used: {(total - free) / 1024**3:.2f} GiB / total={total / 1024**3:.2f} GiB")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.raw:
        prompt_text = args.prompt
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(args.device)
    else:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.think,
        )
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(args.device)

    print(f"- prompt: `{args.prompt}` -> {input_ids.shape[1]} tokens")
    if not args.raw:
        print("- chat-templated prompt:")
        print(f"```\n{prompt_text}\n```")

    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end not in eos_ids:
        eos_ids.append(im_end)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_ids=eos_ids,
    )
    completion_ids = output_ids[0, input_ids.shape[1]:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    print("- completion:")
    print(f"```\n{completion_text}\n```")

if __name__ == "__main__":
    main()
