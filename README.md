# GoogLeNet Summary and Implementation

This repository contains a summary and implementation of the GoogLeNet model created using using [Inception](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) architecture for the ILSVRC 2014 Classification Challenge and ILSVRC 2014 Detection Challenge

## Overview

GoogLeNet is a 22 layer deep CNN created using the Inception CNN archicture. This architecture aims to increase accuracy of neural networks without drastically increasing their computational complexity approximating local sparce structure by dense components and applying dimension reduction sand projections at points that would require lots of computational complexity. GoogLeNet implments 9 Inception Modules.

## Background

CNN models can achieve higher accuracy by creating deeper networks or bigger models, but larger leaps in the field have come from the “synergy of deep architectures and classical computer vision” (see R-CNN algorithm by Girshick et al.

It is also important to create models efficient enough to run on mobile and embedded devices with lower memory and computation budgets.

Standard structure of CNNs are stacked convolutional layers → fully-connected layer(s). Training models on larger datasets generally includes increasing the number of layers, the layer size, and using droupout to address overfitting.

Regions with CNN (Proposed by Girshick et al) utilizes low-level cues then uses CNN classifies based on categories at low-level cues. GoogLeNet has a similar model

Improving deep NN by increasing size increases number of levels and units at each. While it produces higher quality models, it is prone to overfitting and may use significant computing resources.

Above issues can be solved by using more sparsely connected achritectures; this also has firmer theoretical underpinnings (see thing in paper). Unfortunately, modern computers are inefficient in computing non-uniform, sparse data structures.

## Inception Model Explained

Inception works by covering optimal local sparse structure by available dense components (in a CNN). Model creates many clusteres concentrated in single region, covered by a layer of 1x1 convolutions in next layer. To avoid patch-alignment, filter sizes are restrictedt 1x1, 3x3, and 5x5.

1x1 convolutions reduce dimension to remove computational bottle necks and increasing depth and width of of networks with out significant performance penalty

Number of 5x5 convolutions can be expensive. This gets even scarier with pooling layers. Applying dimension reductions and projections when computational requirements would

Inception allows for increased units at each stage without significantly increasing computational complexity. This allows for increasing width and depth of model without too much computational difficulties.

## GoogLeNet Model

GoogLeNet is an impementation of this architecture for ILSVRC14 competition. Input size is 224x224 with and takes RGB colour channels as features. In this implementation, each convolutional, reduction, and projection layer uses rectified linear activation. Network designed so that it can be run on individual devices with limited comp resources and low-memory footprint. About 100 layers are used to construc the 22 layers deep network.

Concern around effective back propagation because of large depth. To circumvent, features produced by layers in middle of network were made more discriminiative by adding auxillary classifiers at intermediate yers resulting in additional regularization.

This was trained with DistBelief distributed ML system on CPU for ILSVRC 2014 Classification Challenge and ILSVRC 2014 Detection Challenge.

Results show that approx optimal sparse structure with reaily available dense building blocks is vaible for improving NN for CV, advantage being significant accuracy without largely increasing computational requirements

The input layer takes parameter of shape 224x224 with RBG colour values. First convolutional layer has a patch size of 7x7, reducing input image size. 2 max pooling layers are used btween inception modules, and 9 total inception modules are implemented. Auxillary classifiers are reduced at the third and sixth Inception layers to reduce overfitting Dropout layer is also used befor linear layer to regularize and prevent overfitting.

## Issues

## Positives

## Resources:

- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) - C. Szegedy et al.
- [Deep Learning: GoogLeNet Explained](https://towardsdatascience.com/deep-learning-googlenet-explained-de8861c82765#:~:text=GoogLeNet%20is%20a%2022%2Dlayer,developed%20by%20researchers%20at%20Google.) - R. Alake
