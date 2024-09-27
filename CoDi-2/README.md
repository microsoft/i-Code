<h1 align="center">CoDi-2: In-Context, Interleaved, and Interactive <br> Any-to-Any Generation</h1>
          <div align="center">
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
<div align="center">
	<span class="author-block"><sup>1</sup>UC Berkeley</span>
	<span class="author-block"><sup>2</sup>Microsoft Azure AI</span>
        <span class="author-block"><sup>3</sup>Zoom</span>
	<span class="author-block"><sup>4</sup>UNC Chapel Hill</span>
	<br>
	<span class="author-block"><small><sup>*</sup> Work done at Microsoft and UNC Chapel Hill. <sup>‡</sup> Work done at Microsoft. <sup>†</sup>Corresponding Authors</span></small>
 </div>

## Introduction 

We present CoDi-2, a versatile and interactive Multi-modal Large Language Model (MLLM) that can follow complex multimodal interleaved instructions, conduct in-context learning (ICL), reason, chat, edit, etc., in an any-to-any input-output modality paradigm. By aligning modalities with language for both encoding and generation, CoDi-2 empowers Large Language Models (LLMs) to not only understand complex modality-interleaved instructions and in-context examples, but also autoregressively generate grounded and coherent multimodal outputs in the continuous feature space. To train CoDi-2, we build a large-scale generation dataset encompassing in-context multi-modal instructions across text, vision, and audio. CoDi-2 demonstrates a wide range of zero-shot capabilities for multimodal generation, such as in-context learning, reasoning, and compositionality of any-to-any modality generation through multi-round interactive conversation. CoDi- 2 surpasses previous domain-specific models on tasks such as subject-driven image generation, vision transformation, and audio editing. CoDi-2 signifies a substantial breakthrough in developing a comprehensive multimodal foundation model adept at interpreting in-context language-vision-audio interleaved instructions and producing multimodal outputs.

<p align="center">
  <img align="middle" width="800" src="assets/teaser.gif"/>
</p>
