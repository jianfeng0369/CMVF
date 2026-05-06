<h1 align="center">
  CMVF: Cross-modal Unregistered Video Fusion via Spatio-Temporal Consistency
</h1>

<p align="center">
  <strong>Cross-modal Unregistered Video Fusion via Spatio-Temporal Consistency</strong>
</p>

<p align="center">
  Jianfeng Ding, Hao Zhang, Zhongyuan Wang, Jinsheng Xiao, Xin Tian, Zhen Han, Jiayi Ma
</p>

<p align="center">
  English | <a href="README_zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/jianfeng0369/CMVF"><img alt="CMVF GitHub" src="https://img.shields.io/badge/CMVF-GitHub-181717?style=for-the-badge&logo=github&logoColor=ffffff"></a>
  <a href="https://doi.org/10.1016/j.inffus.2026.104212"><img alt="CMVF Paper" src="https://img.shields.io/badge/CMVF-Paper-FF6C00?style=for-the-badge&logo=elsevier&logoColor=ffffff"></a>
  <a href="https://github.com/jianfeng0369/VidLLVIP"><img alt="VidLLVIP GitHub" src="https://img.shields.io/badge/VidLLVIP-GitHub-181717?style=for-the-badge&logo=github&logoColor=ffffff"></a>
  <a href="https://huggingface.co/datasets/jianfeng0369/VidLLVIP"><img alt="VidLLVIP Hugging Face Dataset" src="https://img.shields.io/badge/VidLLVIP-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000"></a>
</p>

CMVF is a cross-modal unregistered video fusion method built around spatio-temporal consistency. It is designed for infrared and visible video inputs that may be spatially misaligned, and it performs coarse registration, temporally consistent fusion, and fine registration with cross-modal consistency.

## 📰 News

- 🚀 **2026-05-06**: We released the [VidLLVIP dataset](https://github.com/jianfeng0369/VidLLVIP).
- 🎉 **2026-02-05**: Our multimodal video fusion paper [CMVF](https://doi.org/10.1016/j.inffus.2026.104212) was accepted by *Information Fusion*. The code is available in the [CMVF GitHub repository](https://github.com/jianfeng0369/CMVF).

## Motivation

<p align="center">
  <img src="figs/motivation.png" alt="Motivation of CMVF" width="95%" />
</p>

<p align="center">
  <sub><em><strong>Figure 1.</strong> (a) Cross-modal data acquisition in practical applications. (b) Image fusion primarily utilizes static information from multiple sources, lacking integration of spatio-temporal elements from the raw data. (c) Video fusion effectively combines spatial, temporal, and cross-modal information in an end-to-end manner to produce high-quality, stable, and aligned videos.</em></sub>
</p>

## Overview

<p align="center">
  <img src="figs/overview.png" alt="Overview of CMVF" width="95%" />
</p>

<p align="center">
  <sub><em><strong>Figure 2.</strong> The framework of CMVF comprises three main steps: coarse registration, temporal consistency fusion, and fine registration with cross-modal consistency.</em></sub>
</p>

## Preparation

1. Clone the repository:

```bash
git clone https://github.com/jianfeng0369/CMVF.git
cd CMVF
```

2. Create a conda environment and install the required dependencies:

```bash
conda create -n cmvf python=3.10
conda activate cmvf
pip install -r requirements.txt
```

## Fusion

1. Place images in `data/images` and videos in `data/videos`, and make sure paired infrared and visible inputs share the same file names.

2. Modify the `path_vi`, `path_ir`, and `path_op` variables in `test_*.py` to point to your data.

3. Run image fusion testing:

```bash
python test_images.py
```

4. Run video fusion testing:

```bash
python test_videos.py
```

## Citation

If you find our work useful in your research, please consider citing:

### CMVF

```bibtex
@article{cmvf2026ding,
  title   = {CMVF: Cross-modal unregistered video fusion via spatio-temporal consistency},
  journal = {Information Fusion},
  volume  = {132},
  pages   = {104212},
  year    = {2026},
  issn    = {1566-2535},
  author  = {Jianfeng Ding and Hao Zhang and Zhongyuan Wang and Jinsheng Xiao and Xin Tian and Zhen Han and Jiayi Ma}
}
```

### VidLLVIP

If you use the processed VidLLVIP dataset, registration matrices, or preprocessing pipeline, please also cite VidLLVIP:

```bibtex
@dataset{ding2026vidllvip,
  author  = {Ding, Jianfeng},
  title   = {VidLLVIP: A visible-infrared paired video dataset for low-light vision},
  year    = {2026},
  version = {v1.0.0},
  url     = {https://github.com/jianfeng0369/VidLLVIP}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Contact

If you have any questions, please contact <jianfeng0369@gmail.com>.
