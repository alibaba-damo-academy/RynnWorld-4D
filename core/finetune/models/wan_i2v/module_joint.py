"""
Joint Self-Attention RynnWorld4D Transformer Block and Model.

Modified from: https://github.com/huggingface/diffusers (WanTransformer3DModel)
The base single-branch transformer is from diffusers; this module extends it
with cross-modal joint attention across video, depth, and optical flow branches.

Key difference from the standard RynnWorld4DTransformerBlock:
  - After intra-modal self-attention, a cross-modal attention step lets each branch
    attend to the other two branches using shared K/V projections.
  - Efficient design: each branch has ONE shared K/V (reused across all queries from
    other branches), ONE cross-modal Q, and ONE gated output projection.
  - New params per block = 3*(Q + K + V + Out) = 12 linear layers (vs 18 in naive design).
  - Zero-initialized gating ensures smooth transition from stage1.

Parameter overhead at full scale (dim=5120):
  - Per block: 12 * dim^2 ≈ 315M params
  - Last 10 layers: ~3.15B new params
  - Last 5 layers: ~1.57B new params (recommended for memory-constrained setups)
"""

from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from termcolor import cprint

from diffusers.models.transformers.transformer_wan import (
    WanTransformerBlock, FP32LayerNorm, WanAttention, WanAttnProcessor, WanRotaryPosEmbed
)
from diffusers.models.attention import FeedForward
from diffusers import WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_dispatch import dispatch_attention_fn

try:
    from diffusers.models.lora import apply_lora_scale
except ImportError:
    def apply_lora_scale(attr_name):
        def decorator(func):
            return func
        return decorator


class JointRynnWorld4DTransformerBlock(WanTransformerBlock):
    """
    Transformer block with Joint Cross-Modal Attention across video/depth/flow branches.

    After intra-modal self-attention, each branch attends to the other two branches.
    Design: each branch provides shared K/V for others, and each branch has its own Q
    for querying the concatenated K/V from the other two branches.
    
    Zero-initialized output gates ensure no effect at initialization.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        share_ffn: bool = False,
        enable_joint: bool = True,
        joint_frame_wise: bool = False,
        joint_use_rope: bool = False,
        joint_unidirectional: bool = False,
    ):
        super().__init__(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
        self.share_ffn = share_ffn
        self.enable_joint = enable_joint
        self.joint_frame_wise = joint_frame_wise
        self.joint_use_rope = joint_use_rope
        self.joint_unidirectional = joint_unidirectional
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Independent FFN layers for depth and flow
        if not share_ffn:
            self.ffn_depth = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
            self.ffn_flow = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")

        # --- Depth/Flow branch self-attention (intra-modal) ---
        self.norm1_depth = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1_depth = WanAttention(
            dim=dim, heads=num_heads, dim_head=dim // num_heads, eps=eps,
            cross_attention_dim_head=None, processor=WanAttnProcessor(),
        )
        self.norm1_flow = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1_flow = WanAttention(
            dim=dim, heads=num_heads, dim_head=dim // num_heads, eps=eps,
            cross_attention_dim_head=None, processor=WanAttnProcessor(),
        )

        # --- Joint cross-modal attention (efficient shared K/V design) ---
        if enable_joint:
            # Each branch provides K/V for others to attend to
            self.joint_kv_video = nn.Linear(dim, dim * 2, bias=True)   # K + V from video
            self.joint_kv_depth = nn.Linear(dim, dim * 2, bias=True)   # K + V from depth
            self.joint_kv_flow = nn.Linear(dim, dim * 2, bias=True)    # K + V from flow

            # Each branch has Q for attending to others
            self.joint_q_video = nn.Linear(dim, dim, bias=True)
            self.joint_q_depth = nn.Linear(dim, dim, bias=True)
            self.joint_q_flow = nn.Linear(dim, dim, bias=True)

            # Per-branch alignment LayerNorm: normalize each modality independently to
            # similar mean/variance before computing joint Q/K. This reduces numerical
            # shock when RGB, depth, and flow features have different scales.
            self.joint_align_video = nn.LayerNorm(dim, elementwise_affine=True)
            self.joint_align_depth = nn.LayerNorm(dim, elementwise_affine=True)
            self.joint_align_flow = nn.LayerNorm(dim, elementwise_affine=True)

            # QK normalization
            self.joint_norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
            self.joint_norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

            # Gated output projections (zero-initialized for smooth start)
            self.joint_out_video = nn.Linear(dim, dim, bias=True)
            self.joint_out_depth = nn.Linear(dim, dim, bias=True)
            self.joint_out_flow = nn.Linear(dim, dim, bias=True)

            # Learnable modality embeddings (zero-initialized) added to hidden states
            # before joint cross-modal attention. They act as a soft modality tag so
            # the attention knows whether a token comes from RGB, depth, or flow.
            self.modality_embed_video = nn.Parameter(torch.zeros(1, 1, dim))
            self.modality_embed_depth = nn.Parameter(torch.zeros(1, 1, dim))
            self.modality_embed_flow = nn.Parameter(torch.zeros(1, 1, dim))

            # Learnable gates (initialized to 1.0, NOT zero, to avoid gradient deadlock).
            # Output = joint_out(attn) * gate.tanh(). If BOTH joint_out and gate are zero,
            # their gradients mutually depend on the other being non-zero, so both stay
            # locked at 0 forever (a saddle point). With gate=1 and joint_out=0, the initial
            # output is still 0 (smooth start, stage1 preserved), but joint_out gets a
            # non-zero gradient immediately, unlocking the whole cross-modal attention.
            # This mirrors the standard ControlNet zero-conv design (single zero-init).
            self.joint_gate_video = nn.Parameter(torch.ones(1))
            self.joint_gate_depth = nn.Parameter(torch.ones(1))
            self.joint_gate_flow = nn.Parameter(torch.ones(1))

            # Persistent buffer (not a learnable parameter) that the trainer decays from
            # 1.0 → 0.0 via a cosine schedule during stage3. When 0, the RGB branch's
            # forward path is identical to stage1 (no cross-modal interference).
            self.register_buffer("joint_gate_video_decay", torch.ones(1))

            # Zero-init output projections (smooth start, no cross-modal interference)
            nn.init.zeros_(self.joint_out_video.weight)
            nn.init.zeros_(self.joint_out_video.bias)
            nn.init.zeros_(self.joint_out_depth.weight)
            nn.init.zeros_(self.joint_out_depth.bias)
            nn.init.zeros_(self.joint_out_flow.weight)
            nn.init.zeros_(self.joint_out_flow.bias)

        # --- Cross-attention for depth/flow (text conditioning) ---
        self.attn2_depth = WanAttention(
            dim=dim, heads=num_heads, dim_head=dim // num_heads, eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads, processor=WanAttnProcessor(),
        )
        self.attn2_flow = WanAttention(
            dim=dim, heads=num_heads, dim_head=dim // num_heads, eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads, processor=WanAttnProcessor(),
        )

        # Share K/V projections for text cross-attention across all branches
        self.attn2_depth.to_k = self.attn2.to_k
        self.attn2_depth.to_v = self.attn2.to_v
        self.attn2_depth.norm_k = self.attn2.norm_k
        self.attn2_flow.to_k = self.attn2.to_k
        self.attn2_flow.to_v = self.attn2.to_v
        self.attn2_flow.norm_k = self.attn2.norm_k

        self.norm2_depth = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.norm2_flow = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.norm3_depth = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.norm3_flow = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def _apply_rope(self, x: torch.Tensor, rotary_emb) -> torch.Tensor:
        """Apply 3D RoPE to a tensor of shape [B, N, dim], returning [B, N, dim]."""
        if rotary_emb is None:
            return x
        freqs_cos, freqs_sin = rotary_emb
        x = x.unflatten(2, (self.num_heads, self.head_dim))  # [B, N, H, D]
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(x).flatten(2, 3)  # [B, N, dim]

    def _joint_cross_modal_attention(
        self,
        hidden_states_video: torch.Tensor,
        hidden_states_depth: torch.Tensor,
        hidden_states_flow: torch.Tensor,
        num_frames: int,
        rotary_emb=None,
    ):
        """
        Efficient cross-modal attention: shared K/V per source branch.
        Each branch queries the concatenated K/V from the other two branches.
        If joint_frame_wise=True, attention is restricted to the same frame across modalities.
        If joint_use_rope=True, 3D RoPE is applied to Q/K so cross-modal attention
        respects spatial-temporal position.
        """
        # Per-branch alignment LayerNorm before computing joint Q/K/V.
        # This normalizes RGB/depth/flow features to similar scales, reducing
        # numerical shock in the cross-modal dot-product attention.
        hidden_states_video = self.joint_align_video(hidden_states_video)
        hidden_states_depth = self.joint_align_depth(hidden_states_depth)
        hidden_states_flow = self.joint_align_flow(hidden_states_flow)

        # Compute shared K/V for each branch (used by both other branches)
        kv_v = self.joint_kv_video(hidden_states_video)    # [B, N, 2*dim]
        kv_d = self.joint_kv_depth(hidden_states_depth)
        kv_f = self.joint_kv_flow(hidden_states_flow)

        k_v, v_v = kv_v.chunk(2, dim=-1)  # each [B, N, dim]
        k_d, v_d = kv_d.chunk(2, dim=-1)
        k_f, v_f = kv_f.chunk(2, dim=-1)

        # Compute Q for each branch
        q_video = self.joint_norm_q(self.joint_q_video(hidden_states_video))
        q_depth = self.joint_norm_q(self.joint_q_depth(hidden_states_depth))
        q_flow = self.joint_norm_q(self.joint_q_flow(hidden_states_flow))

        # Normalize K per-branch (RMSNorm is per-token, so this is equivalent to
        # normalizing after concatenation, but allows RoPE to be applied in the
        # correct order: norm -> RoPE, matching WanAttnProcessor).
        k_v = self.joint_norm_k(k_v)
        k_d = self.joint_norm_k(k_d)
        k_f = self.joint_norm_k(k_f)

        # Apply 3D RoPE to Q and K so cross-modal attention respects position.
        # All branches share the same spatial-temporal grid, so the same rotary_emb applies.
        if self.joint_use_rope and rotary_emb is not None:
            q_video = self._apply_rope(q_video, rotary_emb)
            q_depth = self._apply_rope(q_depth, rotary_emb)
            q_flow = self._apply_rope(q_flow, rotary_emb)
            k_v = self._apply_rope(k_v, rotary_emb)
            k_d = self._apply_rope(k_d, rotary_emb)
            k_f = self._apply_rope(k_f, rotary_emb)

        if self.joint_frame_wise and num_frames > 1:
            # Frame-wise cross-modal attention: RGB frame i only attends to
            # depth/flow frame i. This prevents motion blur from cross-time mixing.
            B, N, dim = q_video.shape
            assert N % num_frames == 0, f"Sequence length {N} not divisible by num_frames {num_frames}"
            S = N // num_frames

            def to_per_frame(x):
                # [B, T*S, dim] -> [B*T, S, dim]
                return x.reshape(B, num_frames, S, dim).reshape(B * num_frames, S, dim)

            def from_per_frame(x):
                # [B*T, S, dim] -> [B, T*S, dim]
                return x.reshape(B, num_frames, S, dim).reshape(B, N, dim)

            q_depth_pf = to_per_frame(q_depth)
            q_flow_pf = to_per_frame(q_flow)
            k_v_pf = to_per_frame(k_v)
            v_v_pf = to_per_frame(v_v)

            if self.joint_unidirectional:
                # --- Depth attends to [video] only ---
                out_d = from_per_frame(self._compute_attention(q_depth_pf, k_v_pf, v_v_pf))
                out_d = self.joint_out_depth(out_d)

                # --- Flow attends to [video] only ---
                out_f = from_per_frame(self._compute_attention(q_flow_pf, k_v_pf, v_v_pf))
                out_f = self.joint_out_flow(out_f)

                out_v = None
            else:
                q_video_pf = to_per_frame(q_video)
                k_d_pf = to_per_frame(k_d)
                k_f_pf = to_per_frame(k_f)
                v_d_pf = to_per_frame(v_d)
                v_f_pf = to_per_frame(v_f)

                # --- Video attends to [depth, flow] ---
                k_for_v = torch.cat([k_d_pf, k_f_pf], dim=1)
                v_for_v = torch.cat([v_d_pf, v_f_pf], dim=1)
                out_v = from_per_frame(self._compute_attention(q_video_pf, k_for_v, v_for_v))
                out_v = self.joint_out_video(out_v)

                # --- Depth attends to [video, flow] ---
                k_for_d = torch.cat([k_v_pf, k_f_pf], dim=1)
                v_for_d = torch.cat([v_v_pf, v_f_pf], dim=1)
                out_d = from_per_frame(self._compute_attention(q_depth_pf, k_for_d, v_for_d))
                out_d = self.joint_out_depth(out_d)

                # --- Flow attends to [video, depth] ---
                k_for_f = torch.cat([k_v_pf, k_d_pf], dim=1)
                v_for_f = torch.cat([v_v_pf, v_d_pf], dim=1)
                out_f = from_per_frame(self._compute_attention(q_flow_pf, k_for_f, v_for_f))
                out_f = self.joint_out_flow(out_f)
        else:
            if self.joint_unidirectional:
                # --- Depth attends to [video] only ---
                out_d = self._compute_attention(q_depth, k_v, v_v)
                out_d = self.joint_out_depth(out_d)

                # --- Flow attends to [video] only ---
                out_f = self._compute_attention(q_flow, k_v, v_v)
                out_f = self.joint_out_flow(out_f)

                out_v = None
            else:
                # --- Video attends to [depth, flow] ---
                k_for_v = torch.cat([k_d, k_f], dim=1)  # [B, 2N, dim]
                v_for_v = torch.cat([v_d, v_f], dim=1)

                out_v = self._compute_attention(q_video, k_for_v, v_for_v)
                out_v = self.joint_out_video(out_v)

                # --- Depth attends to [video, flow] ---
                k_for_d = torch.cat([k_v, k_f], dim=1)
                v_for_d = torch.cat([v_v, v_f], dim=1)

                out_d = self._compute_attention(q_depth, k_for_d, v_for_d)
                out_d = self.joint_out_depth(out_d)

                # --- Flow attends to [video, depth] ---
                k_for_f = torch.cat([k_v, k_d], dim=1)
                v_for_f = torch.cat([v_v, v_d], dim=1)

                out_f = self._compute_attention(q_flow, k_for_f, v_for_f)
                out_f = self.joint_out_flow(out_f)

        return out_v, out_d, out_f

    def _compute_attention(self, q, k, v):
        """Standard multi-head attention computation."""
        q = q.unflatten(2, (self.num_heads, self.head_dim))
        k = k.unflatten(2, (self.num_heads, self.head_dim))
        v = v.unflatten(2, (self.num_heads, self.head_dim))

        out = dispatch_attention_fn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        return out.flatten(2, 3).type_as(q)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_depth: torch.Tensor,
        hidden_states_flow: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        num_frames: int = 1,
    ):
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Intra-modal self-attention (each branch attends only within itself)
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        norm_d = (self.norm1_depth(hidden_states_depth.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states_depth)
        attn_output_depth = self.attn1_depth(norm_d, None, None, rotary_emb)
        hidden_states_depth = (hidden_states_depth.float() + attn_output_depth * gate_msa).type_as(hidden_states_depth)

        norm_f = (self.norm1_flow(hidden_states_flow.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states_flow)
        attn_output_flow = self.attn1_flow(norm_f, None, None, rotary_emb)
        hidden_states_flow = (hidden_states_flow.float() + attn_output_flow * gate_msa).type_as(hidden_states_flow)

        # 2. Joint cross-modal attention (each branch attends to the other two)
        if self.enable_joint:
            if self.joint_unidirectional:
                # Unidirectional: only depth/flow receive joint injection,
                # video is the source/teacher and stays unchanged.
                hidden_states_depth = hidden_states_depth + self.modality_embed_depth
                hidden_states_flow = hidden_states_flow + self.modality_embed_flow
                # Note: video is also passed to _joint_cross_modal_attention as the K/V source,
                # but we do NOT add modality_embed_video since video is not modified anyway.

                joint_v, joint_d, joint_f = self._joint_cross_modal_attention(
                    hidden_states, hidden_states_depth, hidden_states_flow, num_frames, rotary_emb
                )
                hidden_states_depth = hidden_states_depth + joint_d * self.joint_gate_depth.tanh()
                hidden_states_flow = hidden_states_flow + joint_f * self.joint_gate_flow.tanh()

                with torch.no_grad():
                    eps = 1e-8
                    self._joint_ratio_video = torch.zeros((), device=hidden_states.device)
                    self._joint_ratio_depth = (joint_d * self.joint_gate_depth.tanh()).float().pow(2).mean().sqrt() / hidden_states_depth.float().pow(2).mean().sqrt().clamp_min(eps)
                    self._joint_ratio_flow  = (joint_f * self.joint_gate_flow.tanh()).float().pow(2).mean().sqrt()  / hidden_states_flow.float().pow(2).mean().sqrt().clamp_min(eps)
            else:
                # Add modality tags so joint attention can distinguish RGB / depth / flow tokens
                hidden_states = hidden_states + self.modality_embed_video * self.joint_gate_video_decay
                hidden_states_depth = hidden_states_depth + self.modality_embed_depth
                hidden_states_flow = hidden_states_flow + self.modality_embed_flow

                joint_v, joint_d, joint_f = self._joint_cross_modal_attention(
                    hidden_states, hidden_states_depth, hidden_states_flow, num_frames, rotary_emb
                )
                hidden_states = hidden_states + joint_v * self.joint_gate_video.tanh() * self.joint_gate_video_decay
                hidden_states_depth = hidden_states_depth + joint_d * self.joint_gate_depth.tanh()
                hidden_states_flow = hidden_states_flow + joint_f * self.joint_gate_flow.tanh()

                # Net contribution ratio = ||joint * gate.tanh()|| / ||hidden_after||.
                # Kept as on-device scalar tensors (no host sync) so this hot path stays fast;
                # the trainer reads .item() at log time only.
                with torch.no_grad():
                    eps = 1e-8
                    self._joint_ratio_video = (joint_v * self.joint_gate_video.tanh() * self.joint_gate_video_decay).float().pow(2).mean().sqrt() / hidden_states.float().pow(2).mean().sqrt().clamp_min(eps)
                    self._joint_ratio_depth = (joint_d * self.joint_gate_depth.tanh()).float().pow(2).mean().sqrt() / hidden_states_depth.float().pow(2).mean().sqrt().clamp_min(eps)
                    self._joint_ratio_flow  = (joint_f * self.joint_gate_flow.tanh()).float().pow(2).mean().sqrt()  / hidden_states_flow.float().pow(2).mean().sqrt().clamp_min(eps)

        # 3. Cross-attention (text conditioning)
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        norm_hidden_states_depth = self.norm2_depth(hidden_states_depth.float()).type_as(hidden_states_depth)
        attn_output_depth = self.attn2_depth(norm_hidden_states_depth, encoder_hidden_states, None, None)
        hidden_states_depth = hidden_states_depth + attn_output_depth

        norm_hidden_states_flow = self.norm2_flow(hidden_states_flow.float()).type_as(hidden_states_flow)
        attn_output_flow = self.attn2_flow(norm_hidden_states_flow, encoder_hidden_states, None, None)
        hidden_states_flow = hidden_states_flow + attn_output_flow

        # 4. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        norm_hidden_states_depth = (self.norm3_depth(hidden_states_depth.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states_depth)
        ff_output = (self.ffn_depth if not self.share_ffn else self.ffn)(norm_hidden_states_depth)
        hidden_states_depth = (hidden_states_depth.float() + ff_output.float() * c_gate_msa).type_as(hidden_states_depth)

        norm_hidden_states_flow = (self.norm3_flow(hidden_states_flow.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states_flow)
        ff_output = (self.ffn_flow if not self.share_ffn else self.ffn)(norm_hidden_states_flow)
        hidden_states_flow = (hidden_states_flow.float() + ff_output.float() * c_gate_msa).type_as(hidden_states_flow)

        return hidden_states, hidden_states_depth, hidden_states_flow


class JointRynnWorld4DTransformer3DModel(WanTransformer3DModel):
    """
    RynnWorld4D Transformer with Joint Self-Attention between video/depth/flow branches.

    Supports loading from a stage1 checkpoint (independent branches) and adding
    joint attention layers for stage2 training.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        share_ffn: bool = False,
        joint_start_layer: int = 0,  # 0-based index where joint attention starts
        joint_end_layer: int = -1,   # 0-based exclusive end; -1 means num_layers
        joint_every_n_layers: int = 1,  # place JA every N layers inside [start, end)
        joint_frame_wise: bool = False,  # restrict cross-modal attention to same-frame tokens
        joint_use_rope: bool = False,  # apply 3D RoPE to Q/K in joint cross-modal attention
        joint_unidirectional: bool = False,  # if True, video is K/V source only; depth/flow only attend to video
    ) -> None:
        super().__init__(
            patch_size, num_attention_heads, attention_head_dim,
            in_channels, out_channels, text_dim, freq_dim, ffn_dim,
            num_layers, cross_attn_norm, qk_norm, eps, image_dim,
            added_kv_proj_dim, rope_max_seq_len, pos_embed_seq_len,
        )
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.patch_embedding_depth = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_flow = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # Build transformer blocks: joint attention enabled according to start/end/every-n pattern
        # Default end_layer=-1 means "until the last layer"; every_n=1 means every layer in range.
        if joint_end_layer < 0:
            joint_end_layer = num_layers
        self.blocks = nn.ModuleList([
            JointRynnWorld4DTransformerBlock(
                inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim,
                share_ffn=share_ffn,
                enable_joint=(
                    (i >= joint_start_layer)
                    and (i < joint_end_layer)
                    and ((i - joint_start_layer) % joint_every_n_layers == 0)
                ),
                joint_frame_wise=joint_frame_wise,
                joint_use_rope=joint_use_rope,
                joint_unidirectional=joint_unidirectional,
            )
            for i in range(num_layers)
        ])

        self.norm_out_depth = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out_depth = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.norm_out_flow = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out_flow = nn.Linear(inner_dim, out_channels * math.prod(patch_size))

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_depth: torch.Tensor,
        hidden_states_flow: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        hidden_states_depth = self.patch_embedding_depth(hidden_states_depth)
        hidden_states_depth = hidden_states_depth.flatten(2).transpose(1, 2)

        hidden_states_flow = self.patch_embedding_flow(hidden_states_flow)
        hidden_states_flow = hidden_states_flow.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states, hidden_states_depth, hidden_states_flow = self._gradient_checkpointing_func(
                    block, hidden_states, hidden_states_depth, hidden_states_flow,
                    encoder_hidden_states, timestep_proj, rotary_emb, post_patch_num_frames
                )
        else:
            for block in self.blocks:
                hidden_states, hidden_states_depth, hidden_states_flow = block(
                    hidden_states, hidden_states_depth, hidden_states_flow,
                    encoder_hidden_states, timestep_proj, rotary_emb, post_patch_num_frames
                )

        # Output norm, projection & unpatchify
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # Video output
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        # Depth output
        hidden_states_depth = (self.norm_out_depth(hidden_states_depth.float()) * (1 + scale) + shift).type_as(hidden_states_depth)
        hidden_states_depth = self.proj_out_depth(hidden_states_depth)
        hidden_states_depth = hidden_states_depth.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states_depth = hidden_states_depth.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output_depth = hidden_states_depth.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        # Flow output
        hidden_states_flow = (self.norm_out_flow(hidden_states_flow.float()) * (1 + scale) + shift).type_as(hidden_states_flow)
        hidden_states_flow = self.proj_out_flow(hidden_states_flow)
        hidden_states_flow = hidden_states_flow.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states_flow = hidden_states_flow.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output_flow = hidden_states_flow.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output, output_depth, output_flow)

        return (
            Transformer2DModelOutput(sample=output),
            Transformer2DModelOutput(sample=output_depth),
            Transformer2DModelOutput(sample=output_flow),
        )

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Allow resuming from checkpoints saved before modality_embed or joint_align
        # were added. These params are zero-initialized, so missing keys are safe to ignore.
        own_keys = set(self.state_dict().keys())
        missing = own_keys - set(state_dict.keys())
        safe_new_prefixes = ("modality_embed", "joint_align", "joint_gate_video_decay")
        if missing and all(any(prefix in k for prefix in safe_new_prefixes) for k in missing):
            cprint(f"⚠️ Loading checkpoint missing {len(missing)} zero-init keys: {sorted(missing)[:5]}...", "yellow")
            return super().load_state_dict(state_dict, strict=False, **kwargs)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load from pretrained Wan model, initializing all 3 branches from the single-branch weights."""
        share_ffn = kwargs.pop("share_ffn", False)
        joint_start_layer = kwargs.pop("joint_start_layer", 0)
        joint_end_layer = kwargs.pop("joint_end_layer", -1)
        joint_every_n_layers = kwargs.pop("joint_every_n_layers", 1)
        joint_frame_wise = kwargs.pop("joint_frame_wise", False)
        joint_use_rope = kwargs.pop("joint_use_rope", False)
        joint_unidirectional = kwargs.pop("joint_unidirectional", False)

        base_model = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config = dict(base_model.config)
        config["share_ffn"] = share_ffn
        config["joint_start_layer"] = joint_start_layer
        config["joint_end_layer"] = joint_end_layer
        config["joint_every_n_layers"] = joint_every_n_layers
        config["joint_frame_wise"] = joint_frame_wise
        config["joint_use_rope"] = joint_use_rope
        config["joint_unidirectional"] = joint_unidirectional
        model = cls(**config)

        base_sd = base_model.state_dict()
        new_sd = model.state_dict()

        for key, value in base_sd.items():
            # Direct match (condition_embedder, rope, etc.)
            if key in new_sd and new_sd[key].shape == value.shape:
                new_sd[key] = value

            # Patch Embedding
            if "patch_embedding." in key:
                new_sd[key.replace("patch_embedding.", "patch_embedding_depth.")] = value.clone()
                new_sd[key.replace("patch_embedding.", "patch_embedding_flow.")] = value.clone()

            # Output Projection
            if "norm_out." in key:
                new_sd[key.replace("norm_out.", "norm_out_depth.")] = value.clone()
                new_sd[key.replace("norm_out.", "norm_out_flow.")] = value.clone()
            if "proj_out." in key:
                new_sd[key.replace("proj_out.", "proj_out_depth.")] = value.clone()
                new_sd[key.replace("proj_out.", "proj_out_flow.")] = value.clone()

            # Transformer Blocks
            if "blocks." in key:
                k_depth = key.replace("attn1.", "attn1_depth.").replace("norm1.", "norm1_depth.") \
                             .replace("attn2.", "attn2_depth.").replace("norm2.", "norm2_depth.") \
                             .replace("norm3.", "norm3_depth.")
                if k_depth in new_sd:
                    new_sd[k_depth] = value.clone()

                k_flow = key.replace("attn1.", "attn1_flow.").replace("norm1.", "norm1_flow.") \
                            .replace("attn2.", "attn2_flow.").replace("norm2.", "norm2_flow.") \
                            .replace("norm3.", "norm3_flow.")
                if k_flow in new_sd:
                    new_sd[k_flow] = value.clone()

                # Independent FFN: clone from RGB FFN
                if ".ffn." in key:
                    k_ffn_depth = key.replace(".ffn.", ".ffn_depth.")
                    k_ffn_flow = key.replace(".ffn.", ".ffn_flow.")
                    if k_ffn_depth in new_sd:
                        new_sd[k_ffn_depth] = value.clone()
                    if k_ffn_flow in new_sd:
                        new_sd[k_ffn_flow] = value.clone()

        model.load_state_dict(new_sd)
        return model
