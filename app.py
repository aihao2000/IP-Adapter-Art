import os

import torch
import spaces

import gradio as gr
import torch
from artistic_portrait.pipeline import ArtisticPortraitXLPipeline
from diffusers import ControlNetModel, DPMSolverMultistepScheduler
from ip_adapter_diffusers.ip_adapter import *

from huggingface_hub import hf_hub_download

style_adapter_path = "models/ip_adapter_art_sdxl_512.pth"
id_adapter_path = "models/pulid_adapter_diffusers_1.1.pth"
if not os.path.exists("models/csd_clip.pth"):
    hf_hub_download(
        repo_id="AisingioroHao0/IP-Adapter-Art",
        filename="csd_clip.pth",
        local_dir="models",
    )
if not os.path.exists(style_adapter_path):
    hf_hub_download(
        repo_id="AisingioroHao0/IP-Adapter-Art",
        filename="ip_adapter_art_sdxl_512.pth",
        local_dir="models",
    )
if not os.path.exists(id_adapter_path):
    hf_hub_download(
        repo_id="AisingioroHao0/IP-Adapter-Art",
        filename="pulid_adapter_diffusers_1.1.pth",
        local_dir="models",
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
sdxl_repo_id = "stabilityai/stable-diffusion-xl-base-1.0"

torch_dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# Load pretrained models.
print("Initializing pipeline...")
controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch_dtype,
).to(device)
pipe = ArtisticPortraitXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch_dtype,
    style_adapter_path=style_adapter_path,
    id_adapter_path=id_adapter_path,
    device=device,
).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)
load_ip_adapter(
    pipe.controlnet,
    "models/ip_adapter_art_sdxl_512.pth",
)

example_inputs = [
    [
        "datasets/test/style_dataset/Abstract D'Oyley.jpg",
        "datasets/test/id_dataset/lifeifei.jpg",
    ],
    [
        "datasets/test/style_dataset/Adam Zyglis.jpg",
        "datasets/test/id_dataset/lecun.jpg",
    ],
    [
        "datasets/test/style_dataset/Diffused lighting.jpg",
        "datasets/test/id_dataset/liuyifei.jpg",
    ],
    [
        "datasets/test/style_dataset/Shirley Hughes.jpg",
        "datasets/test/id_dataset/rihanna.jpg",
    ],
    [
        "datasets/test/style_dataset/Winter.jpg",
        "datasets/test/id_dataset/hinton.jpg",
    ],
]


@spaces.GPU(enable_queue=True)
def generation(
    style_image=None,
    id_image=None,
    pose_image=None,
    prompt="portrait, solo, looking at viewer, best quality, masterpiece",
    negative_prompt="flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, low resolution, partially rendered objects,  deformed or partially rendered eyes, deformed, deformed eyeballs, cross-eyed",
    num_inference_steps=20,
    guidance_scale=7.0,
    style_scale=1.0,
    id_scale=1.0,
    controlnet_scale=0.9,
    seed=42,
    height=1024,
    width=1024,
    artify_contorlnet_scale=0.0,
):
    set_ip_adapter_scale(pipe.controlnet, artify_contorlnet_scale)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=pose_image,
        controlnet_conditioning_scale=controlnet_scale,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        style_image=style_image,
        id_image=id_image,
        generator=torch.Generator(device).manual_seed(seed),
        id_scale=id_scale,
        style_scale=style_scale,
    ).images[0]

    return result


with gr.Blocks(delete_cache=(3600, 3600)) as demo:
    gr.Markdown(
        """
    # Artistic Portrait Generation 0.9: Generate Customized Artistic Portrait through Style Reference Images
    
    **Implementation based on [Art-Adapter](https://github.com/aihao2000/IP-Adapter-Art), [PuLID-Adapter](https://github.com/ToTheBeginning/PuLID), and [Instant Style](https://github.com/instantX-research/InstantStyle).**
    
    ## Basic usage: 
    - Stylized Portrait Generation: Upload the style reference image and ID reference image, and click "Generation" to generate the artistic portrait directly.
    - Text-guided Stylization Generation: Set ID Scale to 0, modify prompt, and then try text-guided stylized image generation through **Art-Adapter**. **(Note that ID image cannot be empty in the current version.)**
    
    _If the style similarity is low, try increasing the Stylize Contorlnet Scale, or set the Controlnet Scale to 0._
    
    ## News
    
    - 2025.3.24: We released Artistic Portrait Generation 0.9.
    """
    )
    with gr.Row():
        with gr.Column():

            with gr.Row():
                style_image = gr.Image(
                    label="Style Reference Image",
                    type="pil",
                )
                id_image = gr.Image(
                    label="ID Reference Image",
                    type="pil",
                )
                pose_image = gr.Image(
                    label="Pose Reference Image",
                    type="pil",
                    value="datasets/test/pose.jpg",
                )
            with gr.Row():
                clear_btn = gr.ClearButton()
                generation_btn = gr.Button("Generation")
            with gr.Row():
                id_scale = gr.Number(label="ID Scale", value=1.0, step=0.1)
                style_scale = gr.Number(label="Style Scale", value=1.0, step=0.1)
                controlnet_scale = gr.Number(
                    label="ControlNet Scale", value=0.9, step=0.1
                )
                stylize_contorlnet_scale = gr.Number(
                    label="Stylize ControlNet Scale", value=0.0, step=0.1
                )
                guidance_scale = gr.Number(label="CFG Scale", value=7.0, step=0.1)
            with gr.Row():
                height = gr.Number(label="Height", step=1, maximum=1024, value=1024)
                width = gr.Number(label="Width", step=1, maximum=1024, value=1024)
                seed = gr.Number(label="Seed", value=42, step=1)
                num_inference_steps = gr.Number(label="Steps", value=20, step=1)
            prompt = gr.Textbox(
                label="Prompt",
                value="portrait, solo, looking at viewer, best quality, masterpiece",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, low resolution, partially rendered objects,  deformed or partially rendered eyes, deformed, deformed eyeballs, cross-eyed",
            )

        with gr.Column():
            output = gr.Image(label="Result", type="pil")
    with gr.Row():
        examples = gr.Examples(
            examples=example_inputs,
            inputs=[style_image, id_image],
            outputs=[
                output,
            ],
            fn=lambda x, y: None,
            cache_examples=False,
        )

    clear_btn.add([style_image, id_image, pose_image, output])

    generation_btn.click(
        generation,
        inputs=[
            style_image,
            id_image,
            pose_image,
            prompt,
            negative_prompt,
            num_inference_steps,
            guidance_scale,
            style_scale,
            id_scale,
            controlnet_scale,
            seed,
            height,
            width,
            stylize_contorlnet_scale,
        ],
        outputs=[output],
        api_name="artistic_portrait_gen",
    )

demo.queue().launch(share=True)
