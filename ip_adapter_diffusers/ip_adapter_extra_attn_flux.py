from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import (
    ImageProjection,
    IPAdapterPlusImageProjection,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from diffusers import FluxPipeline


def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class IPAdapterFluxSingleAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self, cross_attention_dim, hidden_size, scale=1.0, num_text_tokens=512
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.scale = scale

        self.to_q_ip = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(cross_attention_dim)

        self.ip_hidden_states = None
        self.num_text_tokens = 512

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query, key = apply_rope(query, key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        ## ip adapter
        image_hidden_states = self.norm1(hidden_states)
        ip_hidden_states = self.norm2(self.ip_hidden_states)

        ip_query = self.to_q_ip(image_hidden_states)
        ip_query = ip_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ip_key = self.to_k_ip(self.ip_hidden_states)
        ip_value = self.to_v_ip(self.ip_hidden_states)
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_hidden_states = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value,
            dropout_p=0.0,
            is_causal=False,
        )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states += self.scale * ip_hidden_states
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states


class IPAdapterFluxAttnProcessor2_0(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, cross_attention_dim, hidden_size, scale=1.0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.scale = scale

        self.to_q_ip = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(cross_attention_dim)

        self.ip_hidden_states = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )
        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query, key = apply_rope(query, key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # ip adapter
        ip_query = self.to_q_ip(self.norm1(hidden_states))
        ip_query = ip_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ip_hidden_states = self.norm2(self.ip_hidden_states)
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_hidden_states = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value,
            dropout_p=0.0,
            is_causal=False,
        )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states, encoder_hidden_states


def save_ip_adapter(dit, path):
    state_dict = {}
    state_dict["encoder_hid_proj"] = dit.encoder_hid_proj.state_dict()

    for name, module in dit.named_modules():
        if isinstance(module, FluxIPAdapterAttnProcessor2_0) or isinstance(
            module, FluxIPAdapterSingleAttnProcessor2_0
        ):
            state_dict[name] = module.state_dict()

    torch.save(state_dict, path)


def load_ip_adapter(
    dit,
    path=None,
    clip_embeddings_dim=1024,
    cross_attention_dim=3072,
    num_image_text_embeds=8,
    attn_blocks=["single", "double"],
):
    if path is not None:
        state_dict = torch.load(path, map_location="cpu")
        clip_embeddings_dim = state_dict["encoder_hid_proj.image_embeds.weight"].shape[
            1
        ]
        num_image_text_embeds = (
            state_dict["encoder_hid_proj.image_embeds.weight"].shape[0]
            // cross_attention_dim
        )
    dit.encoder_hid_proj = ImageProjection(
        cross_attention_dim=cross_attention_dim,
        image_embed_dim=clip_embeddings_dim,
        num_image_text_embeds=num_image_text_embeds,
    ).to(dit.device, dit.dtype)

    for name, module in dit.named_modules():
        if isinstance(module, Attention):
            if "single" in name:
                if "single" in attn_blocks:
                    module.set_processor(
                        IPAdapterFluxSingleAttnProcessor2_0(
                            hidden_size=module.query_dim,
                            cross_attention_dim=cross_attention_dim,
                        ).to(dit.device, dit.dtype)
                    )
            elif "double" in attn_blocks:
                module.set_processor(
                    IPAdapterFluxAttnProcessor2_0(
                        hidden_size=module.query_dim,
                        cross_attention_dim=cross_attention_dim,
                    ).to(dit.device, dit.dtype)
                )

    if path is not None:
        dit.load_state_dict(state_dict, strict=False)


def set_ip_hidden_states(dit, image_embeds):
    for name, module in dit.named_modules():
        if (
            isinstance(module, IPAdapterFluxSingleAttnProcessor2_0)
            or IPAdapterFluxAttnProcessor2_0
        ):
            module.ip_hidden_states = image_embeds.clone()


def clear_ip_hidden_states(dit):
    for name, module in dit.named_modules():
        if (
            isinstance(module, IPAdapterFluxSingleAttnProcessor2_0)
            or IPAdapterFluxAttnProcessor2_0
        ):
            module.ip_hidden_states = None


def set_ip_adapter_scale(dit, scale=1.0):
    for name, module in dit.named_modules():
        if isinstance(module, IPAdapterFluxSingleAttnProcessor2_0) or isinstance(
            module, IPAdapterFluxAttnProcessor2_0
        ):
            module.scale = scale
