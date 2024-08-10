from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0, Attention
from diffusers.models.embeddings import (
    ImageProjection,
    MultiIPAdapterImageProjection,
    IPAdapterPlusImageProjection,
)
import torch


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
    path,
):
    state_dict = torch.load(path, map_location="cpu")

    if "encoder_hid_proj" in state_dict.keys():
        num_image_text_embeds = 4
        clip_embeddings_dim = state_dict["encoder_hid_proj"][
            "image_projection_layers.0.image_embeds.weight"
        ].shape[-1]
        cross_attention_dim = (
            state_dict["encoder_hid_proj"][
                "image_projection_layers.0.image_embeds.weight"
            ].shape[0]
            // 4
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
        unet.encoder_hid_proj.load_state_dict(state_dict["encoder_hid_proj"])
    else:
        unet.encoder_hid_proj = lambda x: x
        cross_attention_dim = state_dict[
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor"
        ]["to_k_ip.0.weight"].shape[-1]

    unet.config.encoder_hid_dim_type = "ip_image_proj"

    for name, module in unet.named_modules():
        if "attn2" in name and isinstance(module, Attention):
            if not isinstance(module.processor, IPAdapterAttnProcessor2_0):
                module.set_processor(
                    IPAdapterAttnProcessor2_0(
                        hidden_size=module.query_dim,
                        cross_attention_dim=cross_attention_dim,
                    ).to(unet.device, unet.dtype)
                )
            module.processor.load_state_dict(
                state_dict[f"{name}.processor"], strict=False
            )
