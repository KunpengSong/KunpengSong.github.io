---
layout: post
title: "Forward Propagate in CNN"
date: 2019-01-5 18:15:06 
description: "Forward Propagate in CNN"
tag: CNN

---

### Forward Propagate in CNN

Convolutional Neural Network is a efficient tool to handle image recognition problems. It has two processes: forward propagate and backward propagate. This article focus on the **mathematical analysis** of the forward propagate process in CNN. 

## Input Layer 

### Convolution of one filter in input layer

This part focuses on the convolution process of one filter in the input layer. Our image size is 256*256 pixels. The input image of CNN is gray and has only one channel: L channel. So the filters of this layer has depth of 1. For the purpose of clarity, we call this input channel “input **volume**”, and its size is 256\*256\*1. Similarly, the “volume size” of filters in this layer is 3\*3\*1. 

![](/home/kunpeng/Documents/website/KunpengSong.github.io/images/posts/Forward_CNN/Screenshot from 2019-01-24 11-20-42.png)

111

![](//images/posts/Forward_CNN/Screenshot from 2019-01-24 11-20-42.png)

222

![](/images/posts/Forward_CNN/Screenshot from 2019-01-24 11-20-42.png)

333

