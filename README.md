# Multi-Moving Camera Pedestrian Tracking with a New Dataset and Global Link Model
[![](http://img.shields.io/badge/cs.CV-arXiv%3A2302.07676-B31B1B.svg)](###)
[![](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-orange)](https://huggingface.co/datasets/jellyShuang/MMCT)

This repository contains the details of the dataset and the Pytorch implementation of the Baseline Method CrossMOT of the Paper:
[Multi-Moving Camera Pedestrian Tracking with a New Dataset and Global Link Model](https://arxiv.org/abs/2302.07676)

## Abstract
Ensuring driving safety for autonomous vehicles has become increasingly crucial, highlighting the need for systematic tracking of pedestrians on the road. Most vehicles are equipped with visual sensors, however, the large-scale visual dataset from different agents has not been well studied yet. Basically, most of the multi-target multi-camera (MTMC) tracking systems are composed of two modules: single camera tracking (SCT) and inter-camera tracking (ICT). To reliably coordinate between them, MTMC tracking has been a very complicated task, while tracking across multi-moving cameras makes it even more challenging. In this paper, we focus on multi-target multi-moving camera (MTMMC) tracking, which is attracting increasing attention from the research community. Observing there are few datasets for MTMMC tracking, we collect a new dataset, called Multi-Moving Camera Track (MMCT), which contains sequences under various driving scenarios. To address the common prob- lems of identity switch easily faced by most existing SCT trackers, especially for moving cameras due to ego-motion between the camera and targets, a lightweight appearance-free global link model, called Linker, is proposed to mitigate the identity switch by associating two disjoint tracklets of the same target into a complete trajectory within the same camera. Incorporated with Linker, existing SCT trackers generally obtain a significant improvement. Moreover, a strong baseline approach of re- identification (Re-ID) is effectively incorporated to extract robust appearance features under varying surroundings for pedestrian association across moving cameras for ICT, resulting in a much improved MTMMC tracking system, which can constitute a step further towards coordinated mining of multiple moving cameras.

- **<a href="#des"> <u>Dataset Description</u>**</a>
  - **<a href="#str"> <u>Dataset Structure</u>**</a>
  - **<a href="#dow"> <u>Dataset Downloads</u>**</a>

## <a id="des">Dataset Description</a>
We collect data in 12 distinct scenarios, named `'A','B','C',...'L`'. Each scenario may include the interaction of two or three cameras on different cars. For example, scene A includes two sequences of `A-I` and `A-II`. There are 32 sequences in total.

### <a id="str">Dataset Structure</a>
```
MMCT
├── data
│ ├── gps
│ └── labelS
└── images
 ├── 1
 │ ├── A
 │ │ ├── IMG_0098-frag-s1-a-fps5.mp4
 │ │ └── jpg
 │ └── C
 │ ├── IMG_0559-frag-s1-c-fps5.mp4
 │ ├── jpg
 ├── 2
 │ ├── A
 │ │ ├── IMG_0094-frag-s2-a-fps5.mp4
 │ │ ├── jpg
 │ ├── B
 │ │ ├── IMG_2248-frag-s2-b-fps5.mp4
 │ │ ├── jpg
 ...
 ├── 12
 │ ├── A
 │ │ ├── IMG_0104-frag-s12-a-fps5.mp4
 │ │ ├── jpg
 │ ├── B
 │ │ ├── IMG_2254-frag-s12-b-fps5.mp4
 │ │ ├── jpg
 │ └── C
 │ ├── IMG_0569-frag-s12-c-fps5.mp4
 │ ├── jpg
```

### <a id="dow">Dataset Downloads</a>
The whole dataset can be downloaded from [Huggingface](https://huggingface.co/datasets/jellyShuang/MMCT). **Note that, each file needs to unzip by the password. You can decompress each `.zip` file in its folder after sending us (2212534@mail.dhu.edu.cn, ytzhang@dhu.edu.cn) the [LICENSE](https://github.com/shengyuhao/DIVOTrack/blob/main/LICENSE.md). in any format.** 

## <a id="con">Contact</a>
If you have any concerns, please contact [2212534@mail.dhu.edu.cnn](2212534@mail.dhu.edu.cn)


