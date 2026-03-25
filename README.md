<h1 align="center">
  <font color="#1f6feb">CMVF: Cross-modal Unregistered Video Fusion via Spatio-Temporal Consistency</font>
</h1>

<p align="center">
  Jianfeng Ding, Hao Zhang, Zhongyuan Wang, Jinsheng Xiao, Xin Tian, Zhen Han, Jiayi Ma
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.inffus.2026.104212">DOI</a> |
  <a href="https://github.com/jianfeng0369/CMVF">Official Repository</a> |
  <a href="https://www.sciencedirect.com/science/article/pii/S1566253526000916">Paper</a>
</p>

# Motivation

<p align="center">
  <img src="figs/motivation.png" alt="Motivation of CMVF" width="95%" />
</p>
<p align="center">
  <sub><em><strong>Figure 1.</strong> (a) Cross-modal data acquisition in practical applications. (b) Image fusion primarily utilizes static information from multiple sources, lacking integration of spatio-temporal elements from the raw data. (c) Video fusion effectively combines spatial, temporal, and cross-modal information in an end-to-end manner to produce high-quality, stable, and aligned videos.</em></sub>
</p>

# Overview

<p align="center">
  <img src="figs/overview.png" alt="Overview of CMVF" width="95%" />
</p>
<p align="center">
  <sub><em><strong>Figure 2.</strong> The framework of CMVF comprises three main steps: coarse registration, temporal consistency fusion, and fine registration with cross-modal consistency.</em></sub>
</p>

# Prepare

1. Clone the repository:
```
git clone https://github.com/jianfeng0369/CMVF.git
```

2. Create a conda environment and install the required dependencies:
```
conda create -n cmvf python=3.10
conda activate cmvf
pip install -r requirements.txt
```

# Fusion

1. Place images in `data/images` and videos in `data/videos`, and make sure they share the same filenames.

2. Modify the `path_vi`, `path_ir`, and `path_op` variables in `test_*.py` to point to your data.

3. Run the script for image fusion testing:
```
python test_images.py
```

4. Run the script for video fusion testing:
```
python test_videos.py
```

# Citation

If you find our work useful in your research, please consider citing:

```
@article{cmvf2026ding,
title = {CMVF: Cross-modal unregistered video fusion via spatio-temporal consistency},
journal = {Information Fusion},
volume = {132},
pages = {104212},
year = {2026},
issn = {1566-2535},
author = {Jianfeng Ding and Hao Zhang and Zhongyuan Wang and Jinsheng Xiao and Xin Tian and Zhen Han and Jiayi Ma},
}
```

# Contact

If you have any questions, please contact jianfeng0369@gmail.com.