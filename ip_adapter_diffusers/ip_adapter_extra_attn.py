from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import (
    ImageProjection,
    Resampler,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class IPAdapterAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapter for PyTorch 2.0.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or `List[float]`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
        self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError(
                "`scale` should be a list of integers with the same length as `num_tokens`."
            )
        self.scale = scale

        self.to_q_ip = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
                ip_hidden_states = ip_hidden_states[0]

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        ip_query = self.to_q_ip(hidden_states)
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_query = ip_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        current_ip_hidden_states = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def save_ip_adapter(unet, path):
    state_dict = {}
    if (
        hasattr(unet, "encoder_hid_proj")
        and unet.encoder_hid_proj is not None
        and isinstance(unet.encoder_hid_proj, torch.nn.Module)
    ):
        state_dict["encoder_hid_proj"] = unet.encoder_hid_proj.state_dict()

    for name, module in unet.attn_processors.items():
        if isinstance(module, torch.nn.Module):
            state_dict[name] = module.state_dict()

    torch.save(state_dict, path)


def load_ip_adapter(
    unet,
    path=None,
    clip_embeddings_dim=1280,
    cross_attention_dim=2048,
    num_image_text_embeds=4,
):
    if path is None:
        state_dict = None
    else:
        state_dict = torch.load(path, map_location="cpu")
        clip_embeddings_dim = state_dict["encoder_hid_proj"][
            "image_projection_layers.0.image_embeds.weight"
        ].shape[-1]
        num_image_text_embeds = (
            state_dict["encoder_hid_proj"][
                "image_projection_layers.0.image_embeds.weight"
            ].shape[0]
            // cross_attention_dim
        )

    if not hasattr(unet, "encoder_hid_proj") or unet.encoder_hid_proj is None:
        unet.encoder_hid_proj = MultiIPAdapterImageProjection(
            [
                ImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=clip_embeddings_dim,
                    num_image_text_embeds=num_image_text_embeds,
                )
            ]
        ).to(unet.device, unet.dtype)
    if state_dict is not None:
        unet.encoder_hid_proj.load_state_dict(state_dict["encoder_hid_proj"])

    unet.config.encoder_hid_dim_type = "ip_image_proj"
    for name, module in unet.named_modules():
        if "attn2" in name and isinstance(module, Attention):
            if not isinstance(module.processor, IPAdapterAttnProcessor2_0):
                module.set_processor(
                    IPAdapterAttnProcessor2_0(
                        hidden_size=module.query_dim,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                    ).to(unet.device, unet.dtype)
                )
            if state_dict is not None:
                module.processor.load_state_dict(state_dict[f"{name}.processor"])


def set_ip_adapter_scale(unet, scale=1.0):
    for name, module in unet.named_modules():
        if isinstance(module, IPAdapterAttnProcessor2_0):
            module.scale = scale
