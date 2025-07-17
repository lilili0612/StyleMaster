# StyleMaster
<div align="center">

[![CVPR'25](https://img.shields.io/badge/arXiv%20paper-2412.07744-b31b1b)](https://arxiv.org/abs/)&nbsp;
[![project page](https://img.shields.io/badge/Project%20page-StyleMaster-pink)](https://zixuan-ye.github.io/stylemaster)&nbsp;

</div>


**[StyleMaster: Stylize Your Video with Artistic Generation and Translation](https://arxiv.org/abs/)**

![image](assets/stylemaster.gif)


[Zixuan Ye](https://zixuan-ye.github.io/)<sup>1 &dagger;</sup>, [Huijuan Huang](https://openreview.net/profile?id=~Huijuan_Huang1)<sup>2&#9993;</sup>, [Xintao Wang](https://xinntao.github.io/)<sup>2</sup>, [Pengfei Wan](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en)<sup>2</sup>, [Di Zhang](https://openreview.net/profile?id=~Di_ZHANG3)<sup>2</sup>, [Wenhan Luo](https://whluo.github.io/)<sup>1&#9993;</sup>

1 Hong Kong University of Science and Technology  
2 Kuaishou Technology  
† Intern at KwaiVGI, Kuaishou Technology  
✉ Corresponding Author

## TODO
- [x] Code and Weight for T2V Implementation on [Wan-1.4B](https://github.com/Wan-Video/Wan2.1) based on [Diffsynth-Studio](https://github.com/modelscope/DiffSynth-Studio) are avaiable.
- [x] Illusion dataset generation

## Update
- [2025.2] StyleMaster has been accepted by CVPR2025!
- [2024.10] [arXiv](https://arxiv.org/abs/2412.07744) preprint is available.

## Introduction

Welcome to **StyleMaster**! StyleMaster focuses on style control, i.e., generating or translating a video to match the style of a given reference image. StyleMaster preserves local textures and enhance global style representations. Additionally, a motion adapter and gray tile ControlNet are employed to enhance motion quality and provide precise content guidance.

## Features

- **Local Patch Selection**: Overcomes content leakage in style transfer by selecting patches with less similarity to text prompts.
- **Global Style Extraction**: Uses a projection module after CLIP supervised by illusion datasets.
- **Motion Adapter**: Enhances motion quality during inference and helps to enhance the style extent.
- **Gray Tile ControlNet**: Provides accessible yet precise content guidance for video style transfer.
- **High-Quality Video Generation**: Generates videos with high style similarity to the reference image and achieves ideal translation results.

## Illusion Dataset Generation
Please refer to [visual_anagrams/readme.md](visual_anagrams/readme.md) for details.

## Training and Inference on StyleMaster-Wan

please refer to [stylemaster-wan/readme.md](stylemaster-wan/readme.md) for details.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/lilili0612/StyleMaster)

## Citation

```bibtex
@inproceedings{ye2025stylemaster,
  title={Stylemaster: Stylize your video with artistic generation and translation},
  author={Ye, Zixuan and Huang, Huijuan and Wang, Xintao and Wan, Pengfei and Zhang, Di and Luo, Wenhan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2630--2640},
  year={2025}
}
