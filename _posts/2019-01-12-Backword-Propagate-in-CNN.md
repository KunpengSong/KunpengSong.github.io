---
layout: post
title: "Backword Propagate in CNN"
date: 2019-01-5 18:15:06 
description: "Backword Propagate in CNN"
tag: CNN


---

This article focus on the back propagate in CNN and its math foundation.

The following picture shows the last 3 layer of CNN. This CNN has n-1 layers totally. And we will demonstrate the **back propagate** and **weight updating** process of on it. 

![](/images/posts/Backword_CNN/IMG_0603.JPG)

**Important Notations:**

![](/images/posts/Backword_CNN/Screenshot from 2019-01-24 12-11-05.png)

Then the Loss function is:

![](/images/posts/Backword_CNN/Screenshot from 2019-01-24 12-12-43.png)

### Update the weights of **last** layer (layer n)

First, let's update the weights of **last** layer. That is: update all parameters of the 2 filters of layer n. 

![](/images/posts/Backword_CNN/IMG_0603 (1).JPG)

![](/images/posts/Backword_CNN/Screenshot from 2019-01-24 12-16-39.png)

![](/images/posts/Backword_CNN/Screenshot from 2019-01-24 12-17-23.png)

So apply **chain rule** again:

![](/images/posts/Backword_CNN/Screenshot from 2019-01-24 12-18-14.png)

![](/images/posts/Backword_CNN/Screenshot from 2019-01-24 12-19-04.png)

### Update the weights of **inner** layer (eg: layer n-1)

Second, let's update the weights of inner layers (eg: layer n-1). That is: update all parameters of the 3 filters of layer n-1.

![](/images/posts/Backword_CNN/002w2.png)

![](/images/posts/Backword_CNN/Screenshot from 2019-01-24 12-21-26.png)

Instead of stopping at equation ③, we have to continue:

![](/images/posts/Backword_CNN/Screenshot from 2019-01-24 12-23-14.png)

![](/images/posts/Backword_CNN/Screenshot from 2019-01-24 12-24-09.png)

### Convergence 

The network reaches a convergence when the outcome of Loss function is stably less than a determined value



*Disclaimer:*

*This is only a study note. Correctness is not guaranteed*

*Original work. Error correction and Forwarding is welcomed*.

