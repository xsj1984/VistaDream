<h2> 
<a href="https://vistadream-project-page.github.io/" target="_blank">VistaDream: Sampling multiview consistent images for single-view scene reconstruction</a>
</h2>

This is the official PyTorch implementation of the following publication:

> **VistaDream: Sampling multiview consistent images for single-view scene reconstruction**<br/>
> [Haiping Wang](https://hpwang-whu.github.io/), [Yuan Liu](https://liuyuan-pal.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Wenping Wang](https://www.cs.hku.hk/people/academic-staff/wenping), [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm)<br/>
> *ICCV 2025*<br/>
> [**Paper**](https://arxiv.org/abs/2410.16892) | [**Project-page**(with Interactive DEMOs)](https://vistadream-project-page.github.io/) 


## 🔭 Introduction
<p align="center">
<strong>TL;DR: VistaDream is a training-free framework to reconstruct a high-quality 3D scene from a single-view image.</strong>
</p>

<table style="width:100%;">
  <tr>
    <td style="text-align:center; vertical-align:middle;">
      <img src="data/assert/victorian.rgb.png" alt="Image" style="height:200px;">
    </td>
    <td style="text-align:center; vertical-align:middle;">
      <img src="data/assert/victorian.rgb.gif" alt="RGB GIF" style="height:200px;">
    </td>
    <td style="text-align:center; vertical-align:middle;">
      <img src="data/assert/victorian.dpt.gif" alt="Depth GIF" style="height:200px;">
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">Input Image</td>
    <td style="text-align:center;">RGBs of the reconstructed scene</td>
    <td style="text-align:center;">Depths of the reconstructed scene</td>
  </tr>
  <tr>
    <td colspan="3" style="text-align:center;">
      More results and interactive demos are provided in the 
      <a href="https://vistadream-project-page.github.io/" target="_blank">Project Page</a>.
    </td>
  </tr>
</table>

<p align="justify">
  摘要： 本文提出了 VistaDream，一种从单视图图像重建 3D 场景的新型框架。近年来，扩散模型能够从单视图输入图像生成高质量的新视图图像。然而，大多数现有方法仅专注于构建输入图像与生成图像之间的一致性，却忽略了生成图像之间的一致性。
VistaDream 通过两阶段流水线解决了这一问题：
  * 第一阶段：VistaDream 首先通过向外扩展边界并结合估计的深度图，构建一个全局粗粒度的 3D 框架。然后，在这个全局框架上，利用基于扩散的迭代 RGB-D 修复技术生成新视图图像，以填充框架中的空洞。
  * 第二阶段：通过一种无需训练的新型多视图一致性采样（MCS）方法，在扩散模型的反向采样过程中引入多视图一致性约束，进一步增强生成的新视图图像之间的一致性。
实验结果表明，VistaDream 无需训练或微调现有扩散模型，仅使用单视图图像即可实现一致且高质量的新视图合成，性能显著优于基线方法。

</p>

## 🆕 News
- 2025-06-26: VistaDream is accepted by ICCV 2025!
- 2024-10-23: Code, [[project page]](https://vistadream-project-page.github.io/), and [[arXiv paper]](https://arxiv.org/abs/2410.16892) are aviliable.

## 💻 Requirements
The code has been tested on:
- Ubuntu 20.04
- CUDA 12.3
- Python 3.10.12
- Pytorch 2.1.0
- GeForce RTX 4090.

## 🔧 Installation
For complete installation instructions, please see [INSTALL.md](INSTALL.md).

## 🚅 Pretrained model
VistaDream is training-free but utilizes pretrained models of several existing projects.
To download pretrained models for [Fooocus](https://github.com/lllyasviel/Fooocus), [Depth-Pro](https://github.com/apple/ml-depth-pro), 
[OneFormer](https://github.com/SHI-Labs/OneFormer), [SD-LCM](https://github.com/luosiallen/latent-consistency-model), run the following command:
```
bash download_weights.sh
```
The pretrained models of [LLaVA](https://github.com/haotian-liu/LLaVA) and [Stable Diffusion-1.5](https://github.com/CompVis/stable-diffusion) will be automatically downloaded from hugging face on the first running.

## 🔦 Demo (Single-View Generation)
Try VistaDream using the following commands:
```
python demo.py
```
Then, you should obtain:
- ```data/sd_readingroom/scene.pth```: the generated gaussian field;
- ```data/sd_readingroom/video_rgb(dpt).mp4```: the rgb(dpt) renderings from the scene.

## 🔦 Demo (Sparse-View Generation)
To use sparse views as input as [demo_here](https://github.com/WHU-USI3DV/VistaDream/issues/14), we need [Dust3r](https://github.com/naver/dust3r) to first reconstruct the input images to 3D as the scaffold (no zoom-out needed).

First download Dust3r [checkpoints](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) and place it at ```tools/Dust3r/checkpoints``` by the following command:
```
wget -P tools/Dust3r/checkpoints https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```
Then try VistaDream with sparse images as input using the following commands:
```
python demo_sparse.py
```
Then, you should obtain:
- ```data/bedroom/scene.pth```: the generated gaussian field;
- ```data/bedroom/video_rgb(dpt).mp4```: the rgb(dpt) renderings from the scene.

## 🔦 Generate your own scene (Single or Sparse views as input)
If you need to improve the reconstruction quality of your own images, please refer to [INSTRUCT.md](pipe/cfgs/INSTRUCT.md)

To visualize the generated gaussian field, you can use the following script:
```
import torch
from ops.utils import save_ply
scene = torch.load(f'data/vistadream/piano/refine.scene.pth')
save_ply(scene,'gf.ply')
```
and feed the ```gf.ply``` to [SuperSplat](https://playcanvas.com/supersplat/editor) for visualization.

## 🔦 ToDo List
- [x] Early check in generation.
- [x] Support more types of camera trajectory. Please follow [Here](ops/trajs/TRAJECTORY.MD) to define your trajectory. An example is given in this [issue](https://github.com/WHU-USI3DV/VistaDream/issues/11).
- [x] Support sparse-view-input (and no pose needed). An example is given in this [issue](https://github.com/WHU-USI3DV/VistaDream/issues/14).
- [ ] Interactive Demo.

## 🔗 Related Projects
We sincerely thank the excellent open-source projects:
- [Fooocus](https://github.com/lllyasviel/Fooocus) for the wonderful inpainting quality;
- [LLaVA](https://github.com/haotian-liu/LLaVA) for the wonderful image analysis and QA ability;
- [Depth-Pro](https://github.com/apple/ml-depth-pro) for the wonderful monocular metric depth estimation accuracy;
- [OneFormer](https://github.com/SHI-Labs/OneFormer) for the wonderful sky segmentation accuracy;
- [StableDiffusion](https://github.com/CompVis/stable-diffusion) for the wonderful image generation/optimization ability.
