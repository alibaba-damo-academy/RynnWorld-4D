# Modified from: https://github.com/huggingface/diffusers (WanTransformer3DModel)
# Extends the single-branch Wan transformer with independent depth and optical
# flow branches, each initialized from the pretrained RGB branch weights.

from core.finetune.schemas import Wan_Components as Components
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models.transformers.transformer_wan import WanTransformerBlock, FP32LayerNorm, WanAttention, WanAttnProcessor, WanRotaryPosEmbed
from diffusers.models.attention import FeedForward
from diffusers import (
    AutoencoderKLWan,              
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,    
    WanTransformer3DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.configuration_utils import register_to_config

try:
    from diffusers.models.lora import apply_lora_scale
except ImportError:
    def apply_lora_scale(attr_name):
        def decorator(func):
            return func
        return decorator

def patched_wan_time_text_image_embedding_forward(
    self,  # The first argument must be `self`
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    timestep_seq_len: Optional[int] = None,
) -> tuple:
    timestep = self.timesteps_proj(timestep.to(torch.float32))
    if timestep_seq_len is not None:
        timestep = timestep.unflatten(0, (-1, timestep_seq_len))

    with torch.autocast(device_type=timestep.device.type, dtype=torch.float32, enabled=True):
        temb = self.time_embedder(timestep)
        timestep_proj = self.time_proj(self.act_fn(temb))

    temb_casted = temb.type_as(encoder_hidden_states)
    timestep_proj = timestep_proj.type_as(encoder_hidden_states)
    
    encoder_hidden_states = self.text_embedder(encoder_hidden_states)

    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

    return temb_casted, timestep_proj, encoder_hidden_states, encoder_hidden_states_image

def wan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
    control_video_latent: torch.Tensor | None = None,
    is_concat=False,
    null_condition=False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    # hidden_states = self.patch_embedding(hidden_states)
    if is_concat:
        if null_condition:
            hidden_states = self.patch_embedding(hidden_states)
        else:
            hidden_states = torch.cat([hidden_states, control_video_latent], dim=1)
            hidden_states = self.control_patch_embedding(hidden_states)
    else:
        if null_condition:
            hidden_states = self.patch_embedding(hidden_states)
        else:
            hidden_states = self.patch_embedding(hidden_states)
            hidden_states_control = self.control_patch_embedding(control_video_latent)
            hidden_states = hidden_states + hidden_states_control

    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()  # batch_size * seq_len
    else:
        ts_seq_len = None

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    if ts_seq_len is not None:
        # batch_size, seq_len, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )
    else:
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    # 5. Output norm, projection & unpatchify
    if temb.ndim == 3:
        # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        # batch_size, inner_dim
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

class Wan_Components(Components):
    high_noise_model : Any = None

# class RynnWorld4DTransformerBlock(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         ffn_dim: int,
#         num_heads: int,
#         qk_norm: str = "rms_norm_across_heads",
#         cross_attn_norm: bool = False,
#         eps: float = 1e-6,
#         added_kv_proj_dim: int | None = None,
#     ):
#         super().__init__()
        
#         self.video_block = WanTransformerBlock(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
#         self.depth_block = WanTransformerBlock(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
#         self.flow_block = WanTransformerBlock(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)

#         self.video_to_depth_zero = nn.Linear(dim, dim)
#         self.video_to_flow_zero = nn.Linear(dim, dim)
#         self.depth_to_video_zero = nn.Linear(dim, dim)
#         self.flow_to_video_zero = nn.Linear(dim, dim)

#         for layer in [self.video_to_depth_zero, self.video_to_flow_zero, 
#                       self.depth_to_video_zero, self.flow_to_video_zero]:
#             nn.init.zeros_(layer.weight)
#             nn.init.zeros_(layer.bias)

#     def forward(
#         self,
#         hidden_states_video: torch.Tensor,
#         hidden_states_depth: torch.Tensor,
#         hidden_states_flow: torch.Tensor,
#         encoder_hidden_states: torch.Tensor,
#         temb: torch.Tensor,
#         rotary_emb: torch.Tensor,
#     ):
#         h_video = self.video_block(hidden_states_video, encoder_hidden_states, temb, rotary_emb)
#         h_depth = self.depth_block(hidden_states_depth, encoder_hidden_states, temb, rotary_emb)
#         h_flow = self.flow_block(hidden_states_flow, encoder_hidden_states, temb, rotary_emb)
        

#         out_video = h_video + self.depth_to_video_zero(h_depth) + self.flow_to_video_zero(h_flow)
        
#         out_depth = h_depth + self.video_to_depth_zero(h_video)
#         out_flow = h_flow + self.video_to_flow_zero(h_video)
        
#         return out_video, out_depth, out_flow

class RynnWorld4DTransformerBlock(WanTransformerBlock):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        fusion_mode: str = "bidirectional",  # "bidirectional", "unidirectional", "none"
        share_ffn: bool = True,  # True: depth/flow share FFN with RGB; False: independent FFNs
    ):
        super().__init__(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
        self.fusion_mode = fusion_mode
        self.share_ffn = share_ffn

        # Create independent FFN layers for depth and flow if not sharing
        if not share_ffn:
            self.ffn_depth = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
            self.ffn_flow = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        
        # 1. Self-attention
        self.norm1_depth = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1_depth = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )
        self.norm1_flow = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1_flow = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2_depth = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.attn2_flow = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )

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

        # Fusion layers - only create what's needed based on fusion_mode
        if fusion_mode == "bidirectional":
            self.video_to_depth_zero = nn.Linear(dim, dim)
            self.video_to_flow_zero = nn.Linear(dim, dim)
            self.depth_to_video_zero = nn.Linear(dim, dim)
            self.flow_to_video_zero = nn.Linear(dim, dim)
            fusion_layers = [self.video_to_depth_zero, self.video_to_flow_zero, 
                           self.depth_to_video_zero, self.flow_to_video_zero]
        elif fusion_mode == "unidirectional":
            # Only depth/flow → video (no video → depth/flow feedback)
            self.depth_to_video_zero = nn.Linear(dim, dim)
            self.flow_to_video_zero = nn.Linear(dim, dim)
            fusion_layers = [self.depth_to_video_zero, self.flow_to_video_zero]
        else:
            # "none" - no fusion layers
            fusion_layers = []

        for layer in fusion_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_depth: torch.Tensor,
        hidden_states_flow: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ):
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        norm_d = (self.norm1_depth(hidden_states_depth.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states_depth)
        attn_output_depth = self.attn1_depth(norm_d, None, None, rotary_emb)
        hidden_states_depth = (hidden_states_depth.float() + attn_output_depth * gate_msa).type_as(hidden_states_depth)

        norm_f = (self.norm1_flow(hidden_states_flow.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states_flow)
        attn_output_flow = self.attn1_flow(norm_f, None, None, rotary_emb)
        hidden_states_flow = (hidden_states_flow.float() + attn_output_flow * gate_msa).type_as(hidden_states_flow)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        norm_hidden_states_depth = self.norm2_depth(hidden_states_depth.float()).type_as(hidden_states_depth)
        attn_output_depth = self.attn2_depth(norm_hidden_states_depth, encoder_hidden_states, None, None)
        hidden_states_depth = hidden_states_depth + attn_output_depth

        norm_hidden_states_flow = self.norm2_flow(hidden_states_flow.float()).type_as(hidden_states_flow)
        attn_output_flow = self.attn2_flow(norm_hidden_states_flow, encoder_hidden_states, None, None)
        hidden_states_flow = hidden_states_flow + attn_output_flow

        # 3. Fusion (conditional on fusion_mode)
        if self.fusion_mode == "bidirectional":
            hidden_states_raw = hidden_states
            hidden_states = hidden_states + self.depth_to_video_zero(hidden_states_depth) + self.flow_to_video_zero(hidden_states_flow)
            hidden_states_depth = hidden_states_depth + self.video_to_depth_zero(hidden_states_raw)
            hidden_states_flow = hidden_states_flow + self.video_to_flow_zero(hidden_states_raw)
        elif self.fusion_mode == "unidirectional":
            # Only depth/flow → video, no feedback to depth/flow
            hidden_states = hidden_states + self.depth_to_video_zero(hidden_states_depth) + self.flow_to_video_zero(hidden_states_flow)
        # else "none": no fusion, branches stay independent

        # 4. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        norm_hidden_states_depth = (self.norm3_depth(hidden_states_depth.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states_depth
        )
        ff_output = (self.ffn_depth if not self.share_ffn else self.ffn)(norm_hidden_states_depth)
        hidden_states_depth = (hidden_states_depth.float() + ff_output.float() * c_gate_msa).type_as(hidden_states_depth)

        norm_hidden_states_flow = (self.norm3_flow(hidden_states_flow.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states_flow
        )
        ff_output = (self.ffn_flow if not self.share_ffn else self.ffn)(norm_hidden_states_flow)
        hidden_states_flow = (hidden_states_flow.float() + ff_output.float() * c_gate_msa).type_as(hidden_states_flow)


        return hidden_states, hidden_states_depth, hidden_states_flow

class RynnWorld4DTransformer3DModel(WanTransformer3DModel):
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
        fusion_mode: str = "bidirectional",  # "bidirectional", "unidirectional", "none"
        share_ffn: bool = True,  # True: depth/flow share FFN with RGB; False: independent FFNs
    ) -> None:
        super().__init__(
            patch_size,
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            text_dim,
            freq_dim,
            ffn_dim,
            num_layers,
            cross_attn_norm,
            qk_norm,
            eps,
            image_dim,
            added_kv_proj_dim,
            rope_max_seq_len,
            pos_embed_seq_len,
        )
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.patch_embedding_depth = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_flow = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                RynnWorld4DTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim,
                    fusion_mode=fusion_mode,
                    share_ffn=share_ffn,
                )
                for _ in range(num_layers)
            ]
        )
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

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states, hidden_states_depth, hidden_states_flow = self._gradient_checkpointing_func(
                    block, hidden_states, hidden_states_depth, hidden_states_flow, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states, hidden_states_depth, hidden_states_flow = block(hidden_states, hidden_states_depth, hidden_states_flow, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        # video
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        # depth
        hidden_states_depth = (self.norm_out_depth(hidden_states_depth.float()) * (1 + scale) + shift).type_as(hidden_states_depth)
        hidden_states_depth = self.proj_out_depth(hidden_states_depth)

        hidden_states_depth = hidden_states_depth.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states_depth = hidden_states_depth.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output_depth = hidden_states_depth.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        # flow
        hidden_states_flow = (self.norm_out_flow(hidden_states_flow.float()) * (1 + scale) + shift).type_as(hidden_states_flow)
        hidden_states_flow = self.proj_out_flow(hidden_states_flow)

        hidden_states_flow = hidden_states_flow.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states_flow = hidden_states_flow.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output_flow = hidden_states_flow.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output, output_depth, output_flow)

        return Transformer2DModelOutput(sample=output), Transformer2DModelOutput(sample=output_depth), Transformer2DModelOutput(sample=output_flow)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Extract RynnWorld4D-specific kwargs before passing to base model
        fusion_mode = kwargs.pop("fusion_mode", "bidirectional")
        share_ffn = kwargs.pop("share_ffn", True)
        
        base_model = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config = dict(base_model.config)
        config["fusion_mode"] = fusion_mode
        config["share_ffn"] = share_ffn
        model = cls(**config)

        base_sd = base_model.state_dict()
        new_sd = model.state_dict()
        
        for key, value in base_sd.items():
            # 1.condition_embedder, rope
            if key in new_sd and new_sd[key].shape == value.shape:
                new_sd[key] = value

            # 2. Patch Embedding
            if "patch_embedding." in key:
                new_sd[key.replace("patch_embedding.", "patch_embedding_depth.")] = value.clone()
                new_sd[key.replace("patch_embedding.", "patch_embedding_flow.")] = value.clone()
            
            # 3. Output Projection
            if "norm_out." in key:
                new_sd[key.replace("norm_out.", "norm_out_depth.")] = value.clone()
                new_sd[key.replace("norm_out.", "norm_out_flow.")] = value.clone()
            if "proj_out." in key:
                new_sd[key.replace("proj_out.", "proj_out_depth.")] = value.clone()
                new_sd[key.replace("proj_out.", "proj_out_flow.")] = value.clone()

            # 4. Transformer Blocks
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

                # 5. Independent FFN: copy pretrained FFN weights to ffn_depth and ffn_flow
                if ".ffn." in key:
                    k_ffn_depth = key.replace(".ffn.", ".ffn_depth.")
                    k_ffn_flow = key.replace(".ffn.", ".ffn_flow.")
                    if k_ffn_depth in new_sd:
                        new_sd[k_ffn_depth] = value.clone()
                    if k_ffn_flow in new_sd:
                        new_sd[k_ffn_flow] = value.clone()

        model.load_state_dict(new_sd)
        return model


