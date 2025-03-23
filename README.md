---
title: Artistic Portrait Generation
emoji: 🎨
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: true
license: apache-2.0
models:
- AisingioroHao0/IP-Adapter-Art
- guozinan/PuLID
- stabilityai/stable-diffusion-xl-refiner-1.0
- xinsir/controlnet-openpose-sdxl-1.0
---

# IP Adapter Art：

<a href='https://huggingface.co/AisingioroHao0/IP-Adapter-Art'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a><a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a> [![**IP Adapter Art Demo**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kV7q3Gzr8GPG9cChdDQ5ncCx84TYjuu3?usp=sharing)

![image-20240807232402569](./README.assets/main.png)

------

## Introduction

IP Adapter Art is a specialized version that uses a professional style encoder. Its goal is to achieve style control through reference images in the text-to-image diffusion model and solve the problems of instability and incomplete stylization of existing methods. This is a preprint version, and more models and training data coming soon.

## How to use

[![**IP Adapter Art Demo**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kV7q3Gzr8GPG9cChdDQ5ncCx84TYjuu3?usp=sharing)  can be used to conduct experiments directly.

For local experiments, please refer to a [demo](https://github.com/aihao2000/IP-Adapter-Art/blob/main/artistic_portrait_gen.ipynb).

Local experiments require a basic torch environment and dependencies:

```
conda create -n artadapter python=3.10
conda activate artadapter
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install -e .
```

## Comparison with Existing Style Control Methods in Diffusion Models

Evaluation using [StyleBench](https://github.com/open-mmlab/StyleShot) style images. Image quality is evaluated using [improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)

|                       | CLIP Style Similarity | CSD Style Similarity | CLIP Text Alignment | Image Quality | Average   |
| --------------------- | --------------------- | -------------------- | ------------------- | ------------- | --------- |
| DEADiff               | 61.99                 | 43.54                | 20.82               | 60.76         | 46.78     |
| StyleShot             | 63.01                 | 52.40                | 18.93               | 55.54         | 47.47     |
| Instant Style         | 65.39                 | 58.39                | 21.09               | 60.62         | 51.37     |
| **Art-Adapter(ours)** | **67.03**             | **65.02**            | 20.25               | **62.23**     | **53.63** |

![image](./README.assets/comparison_with_existing_methods.jpg)


## Examples of Text-guided Stylized Generation

![image-20240808001612810](./README.assets/more_examples.png)

## Artistic Portrait Generation

### Pipeline 

We built an artistic portrait generation pipeline using Art-Adapter, PuLID, and ControlNet. The structure is shown in the figure below.

![image-20240808001612811](./README.assets/artistic_portrait_generation_pipeline.jpg)

### Examples

![image-20240808001612812](./README.assets/artistic_portrait_generation_examples.jpg)

## Stylize ControlNet Parameter Visualization

![image-20240808001612813](./README.assets/stylize_controlnet_parameter_visualization.jpg)

## Citation

```
@misc{ipadapterart,
  author = {Hao Ai, Xiaosai Zhang},
  title = {IP Adapter Art},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/aihao2000/IP-Adapter-Art}}
}
```

## Acknowledgements

