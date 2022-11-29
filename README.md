# Earthformer: Exploring Space-Time Transformers for Earth System Forecasting

[Zhihan Gao](https://scholar.google.com/citations?user=P6ACUAUAAAAJ&hl=en)<sup>2*</sup>, [Xingjian Shi](https://github.com/sxjscience)<sup>1**</sup>, [Hao Wang](http://www.wanghao.in/)<sup>3</sup>, [Yi Zhu](https://bryanyzhu.github.io/)<sup>1</sup>, [Yuyang Wang](https://scholar.google.com/citations?user=IKUm624AAAAJ&hl=en)<sup>1</sup>, [Mu Li](https://github.com/mli)<sup>1</sup>, [Dit-Yan Yeung](https://scholar.google.com/citations?user=nEsOOx8AAAAJ&hl=en)<sup>2</sup>

<sup>1</sup>Amazon Web Services, <sup>2</sup>Hong Kong University of Science and Technology, <sup>3</sup>Rutgers University

<sup>*</sup>Work done while being an intern at Amazon Web Services. <sup>**</sup>Contact person.

[Paper accepted to NeurIPS2022](https://openreview.net/forum?id=lzZstLVGVGW)

[Code](https://github.com/amazon-science/earth-forecasting-transformer)

[Poster](https://deep-earth.s3.amazonaws.com/papers/earthformer_neurips2022/Earthformer_poster_NeurIPS22.pdf)

## Abstract
Conventionally, Earth system (e.g., weather and climate) forecasting relies on numerical simulation with complex physical models and hence is both expensive in computation and demanding on domain expertise. 
With the explosive growth of spatiotemporal Earth observation data in the past decade, data-driven models that apply Deep Learning (DL) are demonstrating impressive potential for various Earth system forecasting tasks.
The Transformer as an emerging DL architecture, despite its broad success in other domains, has limited adoption in this area.
In this paper, we propose **Earthformer**, a space-time Transformer for Earth system forecasting. Earthformer is based on a generic, flexible and efficient space-time attention block, named **Cuboid Attention**. The idea is to decompose the data into cuboids and apply cuboid-level self-attention in parallel. These cuboids are further connected with a collection of global vectors. We conduct experiments on the MovingMNIST dataset and a newly proposed chaotic *N*-body MNIST dataset to verify the effectiveness of cuboid attention and figure out the best design of Earthformer. Experiments on two real-world benchmarks about precipitation nowcasting and El Niño/Southern Oscillation (ENSO) forecasting show that Earthformer achieves state-of-the-art performance.

## Problem Overview

The Earth is **chaotic, high-dimensional, spatiotemporal** and hence a **complex** system (See the following figure for an example of Earth observation data: an example Vertically Integrated Liquid (VIL) observation sequence from the [Storm EVent ImageRy (SEVIR)](https://sevir.mit.edu/) dataset).
Improving forecasting models for the variabilities of Earth has a **huge socioeconomic impact**. E.g., it can help people take necessary precautions to avoid crises, or better utilize natural resources such as wind and solar energy.

![sevir_example](./figures/sevir_example_len7.png)

We formulate Earth system forecasting as a spatiotemporal sequence forecasting problem. 
The Earth observation data, such as radar echo maps from [NEXRAD](https://www.ncei.noaa.gov/products/radar/next-generation-weather-radar) and climate data from [CIMP6](https://gmd.copernicus.org/articles/9/1937/2016/) , are represented as a spatiotemporal sequence.
Based on these observations, the model predicts the *K*-step-ahead future.

## Method
We propose Earthformer, which is a hierarchical Transformer encoder-decoder based on Cuboid Attention. The input observations are encoded as a hierarchy of hidden states and then decoded to the prediction target.

### Architecture
As illustrated in the following figure, the input sequence has length *T* and the target sequence has length *K*. "x *D*" means to stack *D* cuboid attention blocks with residual connection. "*M* x" means to have *M* layers of hierarchies.

![enc_dec](./figures/hierarchical-encoder-decoder-v1.png)

### Cuboid Attention
We propose the generic cuboid attention layer that involves three steps: "decompose", "attend", and "merge".
In addition, we propose to introduce a collection of *P* global vectors to help cuboids scatter and gather crucial global information.

![cuboid_gif](./figures/cuboid_illustration.gif)

By stacking multiple cuboid attention layers with different choices of "cuboid_size", "strategy" and "shift", we are able to efficiently explore existing and potentially more effective space-time attention.

![cuboid_examples](./figures/cub_pattern_together.png)

The following table lists the configurations of the cuboid attention patterns we explored.
The first row shows the configuration of the generic cuboid attention.
If "shift" or "strategy" is not given, we use shift=(0, 0, 0) and strategy="local" by default.
When stacking multiple cuboid attention layers, each layer will be coupled with layer normalization layers and feed-forward network as in the [Pre-LN Transformer](http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf).

![cuboid_table](./figures/cuboid_table.png)

## Experiments

### MovingMNIST
We follow [Unsupervised Learning of Video Representations using LSTMs (ICML2015)](https://arxiv.org/abs/1502.04681) to use the [public MovingMNIST dataset](https://www.cs.toronto.edu/~nitish/unsupervised_video/).

### *N*-body MNIST
We extend MovingMNIST to a more challenging chaotic *N*-body MNIST by adding long-range and non-linear gravitational interactions among moving digits: 

![gravity_law](./figures/gravity_law.png)

**Access to our *N*-body MNIST dataset**:
1. [Download](https://github.com/amazon-science/earth-forecasting-transformer/blob/main/scripts/datasets/nbody/download_nbody_paper.py) the *N*-body MNIST dataset used in our paper from AWS S3.
2. Generate your custom *N*-body MNIST dataset using our [script](https://github.com/amazon-science/earth-forecasting-transformer/blob/main/scripts/datasets/nbody/generate_nbody_dataset.py) and following the [instructions](https://github.com/amazon-science/earth-forecasting-transformer/blob/main/scripts/datasets/nbody/README.md).

The following figure illustrates the chaos in *N*-body MNIST: the effect of a slight disturbance on the initial velocities is much more significant on *N*-body MNIST than on MovingMNIST. 
The top half are two MovingMNIST sequences, where their initial conditions only slightly differ in the the initial velocities. 
The bottom half are two *N*-body MNIST sequences.
 *N*-body MNIST sequence 1 has exactly the same initial condition as MovingMNIST sequence 1. 
 *N*-body MNIST sequence 2 has exactly the same initial condition as MovingMNIST sequence 2. 
 The final positions of digits in MovingMNIST after 20 steps evolution only slightly differ from each other, while the differences are much more significant in the final frames of *N*-body MNIST sequences.

 ![vis_chaos](./figures/vis_chaos.png)

Earthformer is able to more accurately predict the position of the digits with the help of global vectors.
On the contrary, none of the baseline algorithms that achieved solid performance on MovingMNIST gives the correct and precise position of the digit "0" in the last frame.

![vis_nbody20k_test0](./figures/vis_nbody20k_test0.gif)

### SEVIR
[Storm EVent ImageRy (SEVIR) benchmark](https://sevir.mit.edu/) supports scientific research on multiple meteorological applications including precipitation nowcasting, synthetic radar generation, front detection, etc. 
We adopt SEVIR for benchmarking precipitation nowcasting, i.e., to predict the future VIL up to 60 minutes (12 frames) given 65 minutes context VIL (13 frames).

Checkout the public leaderboard of SEVIR on [Papers With Code](https://paperswithcode.com/sota):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/earthformer-exploring-space-time-transformers/weather-forecasting-on-sevir)](https://paperswithcode.com/sota/weather-forecasting-on-sevir?p=earthformer-exploring-space-time-transformers)

### ICAR-ENSO
Dataset available at [TIANCHI](https://tianchi.aliyun.com/dataset/dataDetail?dataId=98942).

### EarthNet2021
[EarthNet2021](https://www.earthnet.tech/en21/ch-task/) is a dataset and challenge for the task of Earth surface forecasting.

![earthnet2021_example](./figures/EarthNet2021_example.png)

Checkout the [official leaderboard](https://www.earthnet.tech/en21/ch-leaderboard/) of EarthNet2021 and the leaderboard publicly available on [Papers With Code](https://paperswithcode.com/sota):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/earthformer-exploring-space-time-transformers/earth-surface-forecasting-on-earthnet2021-iid)](https://paperswithcode.com/sota/earth-surface-forecasting-on-earthnet2021-iid?p=earthformer-exploring-space-time-transformers)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/earthformer-exploring-space-time-transformers/earth-surface-forecasting-on-earthnet2021-ood)](https://paperswithcode.com/sota/earth-surface-forecasting-on-earthnet2021-ood?p=earthformer-exploring-space-time-transformers)

## BibTeX
```
@inproceedings{gao2022earthformer,
  title={Earthformer: Exploring Space-Time Transformers for Earth System Forecasting},
  author={Gao, Zhihan and Shi, Xingjian and Wang, Hao and Zhu, Yi and Wang, Yuyang and Li, Mu and Yeung, Dit-Yan},
  booktitle={NeurIPS},
  year={2022}
}
```
