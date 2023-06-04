<h1 align="center">CoDi: Any-to-Any Generation via Composable Diffusion</h1>
<div align="center">
  <span class="author-block">
    <a href="https://zinengtang.github.io/">Zineng Tang</a><sup>1*</sup>,</span>
  <span class="author-block">
    <a href="https://ziyi-yang.github.io/">Ziyi Yang</a><sup>2†</sup>,</span>
  <span class="author-block">
    <a href="https://www.microsoft.com/en-us/research/people/chezhu/">Chenguang Zhu</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="https://www.microsoft.com/en-us/research/people/nzeng/">Michael Zeng</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="https://www.cs.unc.edu/~mbansal/">Mohit Bansal</a><sup>1†</sup>
  </span>
</div>
<div align="center">
  <span class="author-block"><sup>1</sup>University of North Carolina at Chapel Hill,</span>
  <span class="author-block"><sup>2</sup>Microsoft Azure Cognitive Services Research</span>
  <span class="author-block"><sup>*</sup> Work done at Microsoft internship and UNC. <sup>†</sup>Corresponding Authors</span>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2305.11846-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2305.11846)  [![githubio](https://img.shields.io/badge/GitHub.io-Project_Page-blue?logo=Github&style=flat-square)](https://codi-gen.github.io/)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/ZinengTang/CoDi)

## Introduction 

We present Composable Diffusion (CoDi), a novel generative model capable of generating any combination of output modalities, such as language, image, video, or audio, from any combination of input modalities. Unlike existing generative AI systems, CoDi can generate multiple modalities in parallel and its input is not limited to a subset of modalities like text or image. Despite the absence of training datasets for many combinations of modalities, we propose to align modalities in both the input and output space. This allows CoDi to freely condition on any input combination and generate any group of modalities, even if they are not present in the training data. CoDi employs a novel composable generation strategy which involves building a shared multimodal space by bridging alignment in the diffusion process, enabling the synchronized generation of intertwined modalities, such as temporally aligned video and audio. Highly customizable and flexible, CoDi achieves strong joint-modality generation quality, and outperforms or is on par with the unimodal state-of-the-art for single-modality synthesis.  

<p align="center">
  <img align="middle" width="800" src="assets/teaser.gif"/>
</p>

## Installation
```
conda create -n CoDi python=3.8 # prepare an environment

pip install torch==1.12.1+cu116 torchaudio==0.12.1+cu116 torchvision==0.13.1+cu116 \
-f https://download.pytorch.org/whl/torch_stable.html # change cuda version higher pytorch version is doable

pip install -r requirements.txt
```

## Inference
Download checkpoints from [Hugginface Model](https://huggingface.co/ZinengTang/CoDi)
More coming soon

Run demo.ipynb

## Citation

If you find our work useful, please consider citing:
```
@article{tang2023any,
  title={Any-to-Any Generation via Composable Diffusion},
  author={Tang, Zineng and Yang, Ziyi and Zhu, Chenguang and Zeng, Michael and Bansal, Mohit},
  journal={arXiv preprint arXiv:2305.11846},
  year={2023}
}
```

## Reference

The code structure is based on [Versatile Diffusion](https://github.com/SHI-Labs/Versatile-Diffusion). The audio diffusion model is based on [AudioLDM](https://github.com/haoheliu/AudioLDM). The video diffusion model is partially based on [Make-A-Video](https://github.com/lucidrains/make-a-video-pytorch).

## Contact

Zineng Tang (zn.tang.terran@gmail.com)
