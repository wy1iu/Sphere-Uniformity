# Learning with Hyperspherical Uniformity

By Weiyang Liu, Rongmei Lin, Zhen Liu, Li Xiong, Bernhard Schölkopf, Adrian Weller

### License 
*Sphere-Uniformity* is released under the MIT License (refer to the LICENSE file for details).

### Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Short Video Introduction](#short-video-introduction)
0. [Requirements](#requirements)
0. [Usage](#usage)

### Introduction
Due to the over-parameterization nature, neural networks are a powerful tool for nonlinear function approximation. In order to achieve good generalization on unseen data, a suitable inductive bias is of great importance for neural networks. One of the most straightforward ways is to regularize the neural network with some additional objectives. Motivated by this, hyperspherical uniformity is proposed as a novel family of relational regularizations that impact the interaction among neurons. We consider several geometrically distinct ways to achieve hyperspherical uniformity.

**Our hyperspherical uniformity is accepted to [AISTATS 2021](https://aistats.org/aistats2021/) and the full paper is available on [arXiv](https://arxiv.org/abs/2103.01649) and [here](https://wyliu.com/papers/LiuAISTATS2021.pdf).**

### Citation
If you find our work useful in your research, please consider to cite:

    @InProceedings{Liu2021SphereUni,
        title={Learning with Hyperspherical Uniformity},
        author={Liu, Weiyang and Lin, Rongmei and Liu, Zhen and Xiong, Li and Schölkopf, Bernhard and Weller, Adrian},
        booktitle={AISTATS},
        year={2021}
    }
   
### Short Video Introduction
We also provide a short video introduction to help interested readers quickly go over our work. Please click the following figure to watch the Youtube video.

[![Uniformity_talk](https://img.youtube.com/vi/YN8tVQb-HyE/0.jpg)](https://youtu.be/YN8tVQb-HyE)

### Requirements
1. `Python 3.7.7` 
2. `TensorFlow 1.14.0`

### Usage

#### Part 1: Clone the repositary
```Shell  
git clone https://github.com/wy1iu/Sphere-Uniformity.git
```
#### Part 2: Download the official CIFAR-100 training and testing data (python version)
```Shell  
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

#### Part 3: Train and test with the following code in different folder. 
```Shell
# run Maximum Hyperspherical Separation
cd mhs
python train.py
```

```Shell
# run  Maximum Hyperspherical Polarization
cd mhp
python train.py
```

```Shell
# run  relaxed Maximum Hyperspherical Polarization
cd r_mhp
python train.py
```

```Shell
# run Minimum Hyperspherical Covering
cd mhc
python train.py
```

```Shell
# run Maximum Gram Determinant
cd mgd
python train.py
```
