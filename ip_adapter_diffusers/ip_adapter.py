from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import ImageProjection, MultiIPAdapterImageProjection
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .resampler import Resampler
from typing import Optional
from diffusers.image_processor import IPAdapterMaskProcessor
import math
import warnings
from pulid.encoders_transformer import IDFormer


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
    attn_blocks=["down", "mid", "up"],
):
    if path is None:
        state_dict = None
    else:
        state_dict = torch.load(path, map_location="cpu")
        clip_embeddings_dim = state_dict["encoder_hid_proj"][
            "image_embeds.weight"
        ].shape[-1]
        num_image_text_embeds = (
            state_dict["encoder_hid_proj"]["image_embeds.weight"].shape[0]
            // cross_attention_dim
        )

    if not hasattr(unet, "encoder_hid_proj") or unet.encoder_hid_proj is None:
        unet.encoder_hid_proj = ImageProjection(
            cross_attention_dim=cross_attention_dim,
            image_embed_dim=clip_embeddings_dim,
            num_image_text_embeds=num_image_text_embeds,
        ).to(unet.device, unet.dtype)
    if state_dict is not None:
        unet.encoder_hid_proj.load_state_dict(state_dict["encoder_hid_proj"])

    for name, module in unet.named_modules():
        if (
            "attn2" in name
            and isinstance(module, Attention)
            and any([attn in name for attn in attn_blocks])
        ):
            if not isinstance(module.processor, IPAttnProcessor2_0):
                module.set_processor(
                    IPAttnProcessor2_0(
                        hidden_size=module.query_dim,
                        cross_attention_dim=cross_attention_dim,
                    ).to(unet.device, unet.dtype)
                )
            if state_dict is not None:
                module.processor.load_state_dict(state_dict[f"{name}.processor"])
            else:
                module.processor.to_k_ip.load_state_dict(module.to_k.state_dict())
                module.processor.to_v_ip.load_state_dict(module.to_v.state_dict())


def parse_clip_embeddings_dim(
    path,
    state_dict,
):
    if "pulid" in path:
        return None
    else:
        return state_dict["encoder_hid_proj"]["image_embeds.weight"].shape[-1]


def parse_num_image_text_embeds(path, state_dict, cross_attention_dim=2048):
    if "pulid" in path:
        return None
    else:
        return (
            state_dict["encoder_hid_proj"]["image_embeds.weight"].shape[0]
            // cross_attention_dim
        )


def parse_encoder_hid_proj_module(
    path=None,
    cross_attention_dim=2048,
    image_embed_dim=None,
    num_image_text_embeds=None,
):
    if "pulid" in path:
        return IDFormer()
    else:
        return ImageProjection(
            cross_attention_dim=cross_attention_dim,
            image_embed_dim=image_embed_dim,
            num_image_text_embeds=num_image_text_embeds,
        )


def load_multi_ip_adapter(
    unet,
    paths=None,
    clip_embeddings_dim=[1280],
    cross_attention_dim=2048,
    num_image_text_embeds=[4],
):
    if paths is None:
        state_dict = None
    else:
        state_dict = [torch.load(path, map_location="cpu") for path in paths]
        clip_embeddings_dim = [
            parse_clip_embeddings_dim(path=single_path, state_dict=single_state_dict)
            for single_path, single_state_dict in zip(paths, state_dict)
        ]
        num_image_text_embeds = [
            parse_num_image_text_embeds(
                path=single_path,
                state_dict=single_state_dict,
                cross_attention_dim=unet.config.cross_attention_dim,
            )
            for single_path, single_state_dict in zip(paths, state_dict)
        ]

    if not hasattr(unet, "encoder_hid_proj") or unet.encoder_hid_proj is None:
        unet.encoder_hid_proj = MultiIPAdapterImageProjection(
            [
                parse_encoder_hid_proj_module(
                    path=single_path,
                    cross_attention_dim=unet.config.cross_attention_dim,
                    image_embed_dim=single_clip_embeddings_dim,
                    num_image_text_embeds=single_num_image_text_embeds,
                ).to(unet.device, unet.dtype)
                for single_path, single_clip_embeddings_dim, single_num_image_text_embeds in zip(
                    paths, clip_embeddings_dim, num_image_text_embeds
                )
            ]
        ).to(unet.device, unet.dtype)

    if state_dict is not None:
        for single_encoder_hid_proj, single_state_dict in zip(
            unet.encoder_hid_proj.image_projection_layers, state_dict
        ):
            single_encoder_hid_proj.load_state_dict(
                single_state_dict["encoder_hid_proj"]
            )

    for name, module in unet.named_modules():
        if "attn2" in name and isinstance(module, Attention):
            if not isinstance(module.processor, MultiIPAttnProcessor2_0):
                module.set_processor(
                    MultiIPAttnProcessor2_0(
                        hidden_size=module.query_dim,
                        cross_attention_dim=unet.config.cross_attention_dim,
                        num_tokens=num_image_text_embeds,
                    ).to(unet.device, unet.dtype)
                )
            if state_dict is not None:
                for (
                    to_k_ip,
                    to_v_ip,
                    single_state_dict,
                ) in zip(
                    module.processor.to_k_ip,
                    module.processor.to_v_ip,
                    state_dict,
                ):
                    if f"{name}.processor" in single_state_dict.keys():
                        to_k_ip.weight = nn.Parameter(
                            single_state_dict[f"{name}.processor"]["to_k_ip.weight"]
                        )
                        to_v_ip.weight = nn.Parameter(
                            single_state_dict[f"{name}.processor"]["to_v_ip.weight"]
                        )
            module.processor = module.processor.to(unet.device, unet.dtype)


def load_ip_adapter_plus(
    unet,
    path=None,
    embed_dims=1664,
    depth=4,
    dim_head=64,
    heads=12,
    num_queries=32,
    ff_mult=4,
    attn_blocks=["down", "mid", "up"],
):
    if path is not None:
        state_dict = torch.load(path)
    else:
        state_dict = None
    if not hasattr(unet, "encoder_hid_proj") or unet.encoder_hid_proj is None:
        unet.encoder_hid_proj = Resampler(
            dim=unet.config.cross_attention_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_queries,
            embedding_dim=embed_dims,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=ff_mult,
        ).to(unet.device, unet.dtype)
    if state_dict is not None:
        unet.encoder_hid_proj.load_state_dict(state_dict["encoder_hid_proj"])

    for name, module in unet.named_modules():
        if (
            "attn2" in name
            and isinstance(module, Attention)
            and any([attn in name for attn in attn_blocks])
        ):
            if not isinstance(module.processor, IPAttnProcessor2_0):
                module.set_processor(
                    IPAttnProcessor2_0(
                        hidden_size=module.query_dim,
                        cross_attention_dim=unet.config.cross_attention_dim,
                    ).to(unet.device, unet.dtype)
                )
            if state_dict is not None and f"{name}.processor" in state_dict.keys():
                module.processor.load_state_dict(state_dict[f"{name}.processor"])
            else:
                module.processor.to_k_ip.load_state_dict(module.to_k.state_dict())
                module.processor.to_v_ip.load_state_dict(module.to_v.state_dict())


def set_ip_hidden_states(unet, image_embeds):
    for name, module in unet.attn_processors.items():
        if isinstance(module, IPAttnProcessor2_0) or isinstance(
            module, MultiIPAttnProcessor2_0
        ):
            module.ip_hidden_states = image_embeds.clone()


def set_multi_ip_hidden_states(unet, image_embeds):
    for name, module in unet.attn_processors.items():
        if isinstance(module, IPAttnProcessor2_0) or isinstance(
            module, MultiIPAttnProcessor2_0
        ):
            module.ip_hidden_states = image_embeds


def set_multi_ip_attn_masks(unet, attn_masks):
    for name, module in unet.attn_processors.items():
        if isinstance(module, IPAttnProcessor2_0) or isinstance(
            module, MultiIPAttnProcessor2_0
        ):
            module.ip_hidden_states = attn_masks


def clear_ip_hidden_states(model):
    for name, module in model.named_modules():
        if isinstance(module, IPAttnProcessor2_0):
            module.ip_hidden_states = None


def set_ip_adapter_scale(unet, scale=1.0, attn_blocks=["down", "mid", "up"]):
    for name, module in unet.named_modules():
        if isinstance(module, IPAttnProcessor2_0) and any(
            tarhet_module in name for tarhet_module in attn_blocks
        ):
            module.scale = scale


def downsample(
    mask: torch.Tensor, batch_size: int, num_queries: int, value_embed_dim: int
):
    """
    Downsamples the provided mask tensor to match the expected dimensions for scaled dot-product attention. If the
    aspect ratio of the mask does not match the aspect ratio of the output image, a warning is issued.

    Args:
        mask (`torch.Tensor`):
            The input mask tensor generated with `IPAdapterMaskProcessor.preprocess()`.
        batch_size (`int`):
            The batch size.
        num_queries (`int`):
            The number of queries.
        value_embed_dim (`int`):
            The dimensionality of the value embeddings.

    Returns:
        `torch.Tensor`:
            The downsampled mask tensor.

    """
    o_h = mask.shape[2]
    o_w = mask.shape[3]
    ratio = o_w / o_h
    mask_h = int(math.sqrt(num_queries / ratio))
    mask_h = int(mask_h) + int((num_queries % int(mask_h)) != 0)
    mask_w = num_queries // mask_h

    mask_downsample = F.interpolate(mask, size=(mask_h, mask_w), mode="bicubic")

    # Repeat batch_size times
    if mask_downsample.shape[0] < batch_size:
        mask_downsample = mask_downsample.repeat(batch_size, 1, 1, 1)

    mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1)

    downsampled_area = mask_h * mask_w
    # If the output image and the mask do not have the same aspect ratio, tensor shapes will not match
    # Pad tensor if downsampled_mask.shape[1] is smaller than num_queries
    if downsampled_area < num_queries:
        warnings.warn(
            "The aspect ratio of the mask does not match the aspect ratio of the output image. "
            "Please update your masks or adjust the output size for optimal performance.",
            UserWarning,
        )
        mask_downsample = F.pad(
            mask_downsample, (0, num_queries - mask_downsample.shape[1]), value=0.0
        )
    # Discard last embeddings if downsampled_mask.shape[1] is bigger than num_queries
    if downsampled_area > num_queries:
        warnings.warn(
            "The aspect ratio of the mask does not match the aspect ratio of the output image. "
            "Please update your masks or adjust the output size for optimal performance.",
            UserWarning,
        )
        mask_downsample = mask_downsample[:, :num_queries]

    # Repeat last dimension to match SDPA output shape
    mask_downsample = mask_downsample.view(
        mask_downsample.shape[0], mask_downsample.shape[1], 1
    ).repeat(1, 1, value_embed_dim)

    return mask_downsample


class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        scale=1.0,
        num_tokens=4,
        use_align_sem_and_layout_loss=False,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.ip_hidden_states = None

        self.use_align_sem_and_layout_loss = use_align_sem_and_layout_loss
        if self.use_align_sem_and_layout_loss:
            self.align_sem_loss = None
            self.align_layout_loss = None
            self.cache_query = None
            self.cache_attn_weights = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        ip_adapter_masks: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

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

        if attn.norm_cross:
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
        if self.use_align_sem_and_layout_loss:
            if self.cache_query is None:
                self.cache_query = query.clone().detach()
                self.cache_attn_weights = (key @ query.transpose(-2, -1)) / math.sqrt(
                    query.size(-1)
                )
                self.cache_attn_weights = torch.softmax(self.cache_attn_weights, dim=-1)
            else:
                self.attn_weights = (key @ query.transpose(-2, -1)) / math.sqrt(
                    query.size(-1)
                )
                self.query = query
                self.attn_weights = torch.softmax(self.attn_weights, dim=-1)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if self.scale != 0.0:
            # for ip-adapter
            ip_key = self.to_k_ip(self.ip_hidden_states).to(dtype=query.dtype)
            ip_value = self.to_v_ip(self.ip_hidden_states).to(dtype=query.dtype)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            # with torch.no_grad():
            #     self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            # print(self.attn_map.shape)

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            if ip_adapter_masks is not None:
                mask_downsample = downsample(
                    ip_adapter_masks,
                    batch_size,
                    ip_hidden_states.shape[1],
                    ip_hidden_states.shape[2],
                )

                mask_downsample = mask_downsample.to(
                    dtype=query.dtype, device=query.device
                )

                ip_hidden_states = ip_hidden_states * mask_downsample

            hidden_states = hidden_states + self.scale * ip_hidden_states

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


def set_ortho(unet, ortho):
    for name, module in unet.attn_processors.items():
        if isinstance(module, IPAttnProcessor2_0) or isinstance(
            module, MultiIPAttnProcessor2_0
        ):
            module.ortho = ortho


def set_num_zero(unet, num_zero):
    for name, module in unet.attn_processors.items():
        if isinstance(module, IPAttnProcessor2_0) or isinstance(
            module, MultiIPAttnProcessor2_0
        ):
            module.num_zero = num_zero


class MultiIPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.

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

        self.to_k_ip = nn.ModuleList(
            [
                nn.Linear(cross_attention_dim, hidden_size, bias=False)
                for _ in range(len(num_tokens))
            ]
        )
        self.to_v_ip = nn.ModuleList(
            [
                nn.Linear(cross_attention_dim, hidden_size, bias=False)
                for _ in range(len(num_tokens))
            ]
        )
        self.ip_hidden_states = None
        self.num_zero = [None] * (len(num_tokens))
        self.ortho = [None] * len(num_tokens)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        ip_hidden_states = self.ip_hidden_states

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

        if ip_adapter_masks is not None:
            if (
                not isinstance(ip_adapter_masks, torch.Tensor)
                or ip_adapter_masks.ndim != 4
            ):
                raise ValueError(
                    " ip_adapter_mask should be a tensor with shape [num_ip_adapter, 1, height, width]."
                    " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                )
            if len(ip_adapter_masks) != len(self.scale):
                raise ValueError(
                    f"Number of ip_adapter_masks ({len(ip_adapter_masks)}) must match number of IP-Adapters ({len(self.scale)})"
                )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        # for ip-adapter
        for (
            current_ip_hidden_states,
            scale,
            to_k_ip,
            to_v_ip,
            mask,
            num_zero,
            ortho,
        ) in zip(
            ip_hidden_states,
            self.scale,
            self.to_k_ip,
            self.to_v_ip,
            ip_adapter_masks,
            self.num_zero,
            self.ortho,
        ):
            if scale == 0:
                continue
            if num_zero is not None:
                zero_tensor = torch.zeros(
                    (
                        current_ip_hidden_states.size(0),
                        num_zero,
                        current_ip_hidden_states.size(-1),
                    ),
                    dtype=current_ip_hidden_states.dtype,
                    device=current_ip_hidden_states.device,
                )
                current_ip_hidden_states = torch.concat(
                    [current_ip_hidden_states, zero_tensor], dim=1
                )
            ip_key = to_k_ip(current_ip_hidden_states)
            ip_value = to_v_ip(current_ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            current_ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

            if mask is not None:
                mask_downsample = IPAdapterMaskProcessor.downsample(
                    mask,
                    batch_size,
                    current_ip_hidden_states.shape[1],
                    current_ip_hidden_states.shape[2],
                )

                mask_downsample = mask_downsample.to(
                    dtype=query.dtype, device=query.device
                )

                current_ip_hidden_states = current_ip_hidden_states * mask_downsample
            if ortho is None:
                hidden_states = hidden_states + scale * current_ip_hidden_states
            elif ortho == "ortho":
                orig_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                current_ip_hidden_states = current_ip_hidden_states.to(torch.float32)
                projection = (
                    torch.sum(
                        (hidden_states * current_ip_hidden_states), dim=-2, keepdim=True
                    )
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = current_ip_hidden_states - projection
                hidden_states = hidden_states + current_ip_hidden_states * orthogonal
                hidden_states = hidden_states.to(orig_dtype)
            elif ortho == "ortho_v2":
                orig_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                current_ip_hidden_states = current_ip_hidden_states.to(torch.float32)
                attn_map = query @ ip_key.transpose(-2, -1)
                attn_mean = attn_map.softmax(dim=-1).mean(dim=1)
                attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)
                projection = (
                    torch.sum(
                        (hidden_states * current_ip_hidden_states), dim=-2, keepdim=True
                    )
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = current_ip_hidden_states + (attn_mean - 1) * projection
                hidden_states = hidden_states + current_ip_hidden_states * orthogonal
                hidden_states = hidden_states.to(orig_dtype)
            else:
                raise ValueError(f"{ortho} not supported")

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
