# i-Code: An Integrative and Composable Multimodal Learning Framework (AAAI 2023)
**A multimodal foundation model for vision, language and speech understanding**

## Introduction
We present i-Code, a self-supervised pretraining framework where users may flexibly combine the modalities of vision, speech, and language into unified and general-purpose vector representations. In this framework, data from each modality are first given to pretrained single-modality encoders. The encoder outputs are then integrated with a multimodal fusion network, which uses novel attention mechanisms and other architectural innovations to effectively combine information from the different modalities. The entire system is pretrained end-to-end with new objectives including masked modality unit modeling and cross-modality contrastive learning.The i-Code framework can dynamically process single, dual, and triple-modality data during training and inference, flexibly projecting different combinations of modalities into a single representation space. More details can be found in the paper:

## Citation

If you find our work useful, please consider citing:
```
@inproceedings{yang2022code,
  title={i-code: An integrative and composable multimodal learning framework},
  author={Yang, Ziyi and Fang, Yuwei and Zhu, Chenguang and Pryzant, Reid and Chen, Dongdong and Shi, Yu and Xu, Yichong and Qian, Yao and Gao, Mei and Chen, Yi-Ling and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

[i-Code: An Integrative and Composable Multimodal Learning Framework](https://arxiv.org/abs/2205.01818), AAAI 2023




