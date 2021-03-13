# Diverse Structure Inpainting

## Introduction
This repository is for the CVPR 2021 paper, "Generating Diverse Structure for Image Inpainting with Hierarchical VQ-VAE".

If our method is useful for your research, please consider citing.

![Introduction](./intro.png)

## Installation
This code was tested with TensorFlow 1.12.0, CUDA 9.0, Python 3.6 and Ubuntu 16.04

Clone this repository：
```
git clone https://github.com/USTC-JialunPeng/Diverse-Structure-Inpainting.git
```



## Pre-trained Models
Download the pre-trained models using the following links and put them under `model_logs/` directory.

* `center_mask model`: [CelebA-HQ_center](https://drive.google.com/drive/folders/14Vskk15KUw6kYVkbyCJZw-7PUVhAiYqT) | [Places2_center](https://drive.google.com/drive/folders/1Dwi3HYC8ZDcqZvAnMQQUhMSKWOwkBTLJ) | [ImageNet_center](https://drive.google.com/drive/folders/1UanB-Yi4UkEma7tEsykjKKCKziS5Mb2Z)
* `random_mask model`: [CelebA-HQ_random](https://drive.google.com/drive/folders/1jLGVwWREwfGaKEzsr8f4IUqCCFANkFvG) | [Places2_random](https://drive.google.com/drive/folders/1h6tU-2P1j2DFAD42VntFS7XsKNRBI7__) | [ImageNet_random](https://drive.google.com/drive/folders/1ZNh9vjZGevCjUg-mF08pT6L3KZLo8MTL)

The **center_mask models** are trained with images of 256x256 resolution with center 128x128 holes. The **random_mask models** are trained with random regular and irregular holes.
