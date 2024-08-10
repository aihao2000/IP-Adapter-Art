# IP Adapter Art：

<a href='https://huggingface.co/AisingioroHao0/IP-Adapter-Art'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a><a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a> [![**IP Adapter Art Demo**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kV7q3Gzr8GPG9cChdDQ5ncCx84TYjuu3?usp=sharing)

![image-20240807232402569](./README.assets/main.png)

------

## Introduction

IP Adapter Art is a specialized version that uses a professional style encoder. Its goal is to achieve style control through reference images in the text-to-image diffusion model and solve the problems of instability and incomplete stylization of existing methods. This is a preprint version, and more models and training data coming soon.

## How to use

[![**IP Adapter Art Demo**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kV7q3Gzr8GPG9cChdDQ5ncCx84TYjuu3?usp=sharing)  can be used to conduct experiments directly.

For local experiments, please refer to a [demo](https://github.com/aihao2000/IP-Adapter-Artist/blob/main/ip_adapter_art_sdxl_demo.ipynb).

Local experiments require a basic torch environment and dependencies:

```
pip install diffusers
pip install transformers
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/aihao2000/IP-Adapter-Art.git
```

## More Examples

![image-20240808001612810](./README.assets/more_examples.png)


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