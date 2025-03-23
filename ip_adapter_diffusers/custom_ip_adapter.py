from .custom_cross_attention_processor import DecoupledCrossAttnProcessor2_0
import torch
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0, Attention


def load_custom_ip_adapter(
    unet,
    path=None,
    blocks="full",
    Custom_Attn_Type=DecoupledCrossAttnProcessor2_0,
    cross_attention_dim=2048,
    Image_Proj_Type=None,
):
    if path is None:
        state_dict = None
    else:
        state_dict = torch.load(path, map_location="cpu")

    # unet.config.encoder_hid_dim_type = "ip_image_proj"
    # if Image_Proj_Type is None:
    #     unet.encoder_hid_proj = torch.nn.Identity()
    #     unet.encoder_hid_proj.image_projection_layers = torch.nn.ModuleList(
    #         [torch.nn.Identity()]
    #     )

    for name, module in unet.named_modules():
        if "attn2" in name and isinstance(module, Attention):
            if blocks == "midup" and "mid" not in name and "up" not in name:
                continue
            if not isinstance(module.processor, torch.nn.Module):
                module.set_processor(
                    Custom_Attn_Type(
                        hidden_size=module.query_dim,
                        cross_attention_dim=cross_attention_dim,
                    ).to(unet.device, unet.dtype)
                )
            if state_dict is not None:
                module.processor.load_state_dict(state_dict[f"{name}.processor"])
            else:
                if hasattr(module.processor, "to_q_ip"):
                    torch.nn.init.kaiming_normal_(module.processor.to_q_ip.weight)
                torch.nn.init.kaiming_normal_(module.processor.to_k_ip.weight)
                torch.nn.init.kaiming_normal_(module.processor.to_v_ip.weight)


def save_custom_ip_adapter(unet, path):
    state_dict = {}
    for name, module in unet.attn_processors.items():
        if isinstance(module, torch.nn.Module):
            state_dict[name] = module.state_dict()

    torch.save(state_dict, path)


def set_scale(unet, scale):
    for name, module in unet.attn_processors.items():
        if isinstance(module, torch.nn.Module):
            module.scale = scale
