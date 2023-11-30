<h1 class="title is-1 publication-title">CoDi-2: In-Context, Interleaved, and Interactive<br> Any-to-Any Generation</h1>
          <div class="is-size-5 publication-authors">
            <span class="author-block">
              <a href="https://zinengtang.github.io/">Zineng Tang</a><sup>1,4*</sup>,</span>
            <span class="author-block">
              <a href="https://ziyi-yang.github.io/">Ziyi Yang</a><sup>2†</sup>,</span>
            <span class="author-block">
			<span class="author-block">
			  <a href="https://www.microsoft.com/en-us/research/people/mkhademi/">Mahmoud Khademi</a><sup>3</sup>,</span>
			<span class="author-block">
			<span class="author-block">
			  <a href="https://nlp-yang.github.io/">Yang Liu</a><sup>2</sup>,</span>
			<span class="author-block">
              <a href="https://scholar.google.com/citations?user=1b2kKWoAAAAJ&hl=en">Chenguang Zhu</a><sup>3‡</sup>,
            </span>
            <span class="author-block">
              <a href="https://www.cs.unc.edu/~mbansal/">Mohit Bansal</a><sup>4†</sup>
            </span>
          </div>
<div class="is-size-5 publication-authors">
	<span class="author-block"><sup>1</sup>UC Berkeley</span>
	<span class="author-block"><sup>2</sup>Microsoft Azure AI</span>
        <span class="author-block"><sup>3</sup>Zoom</span>
	<span class="author-block"><sup>4</sup>UNC Chapel Hill</span>
	<span class="author-block"><small><sup>*</sup> Work done at Microsoft and UNC Chapel Hill. <sup>‡</sup> Work done at Microsoft. <sup>†</sup>Corresponding Authors</span></small>
 </div>

[![arXiv](https://img.shields.io/badge/arXiv-2305.11846-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2305.11846)  [![githubio](https://img.shields.io/badge/GitHub.io-Project_Page-blue?logo=Github&style=flat-square)](https://codi-gen.github.io/)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/ZinengTang/CoDi)

## Introduction 

We present Composable Diffusion (CoDi), a novel generative model capable of generating any combination of output modalities, such as language, image, video, or audio, from any combination of input modalities. Unlike existing generative AI systems, CoDi can generate multiple modalities in parallel and its input is not limited to a subset of modalities like text or image. Despite the absence of training datasets for many combinations of modalities, we propose to align modalities in both the input and output space. This allows CoDi to freely condition on any input combination and generate any group of modalities, even if they are not present in the training data. CoDi employs a novel composable generation strategy which involves building a shared multimodal space by bridging alignment in the diffusion process, enabling the synchronized generation of intertwined modalities, such as temporally aligned video and audio. Highly customizable and flexible, CoDi achieves strong joint-modality generation quality, and outperforms or is on par with the unimodal state-of-the-art for single-modality synthesis.  

<p align="center">
  <img align="middle" width="800" src="assets/teaser.gif"/>
</p>
