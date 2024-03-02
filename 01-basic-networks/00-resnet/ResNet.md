# ResNet

Paper Name: Deep Residual Learning for Image Recognition
URL: https://arxiv.org/abs/1512.03385.pdf

```
ResNet, short for Residual Networks, is a classic neural network used as a backbone for many computer vision tasks. This model was proposed by Kaiming He et al. in their 2015 paper, "Deep Residual Learning for Image Recognition".

The key innovation of ResNet is the introduction of "skip connections" or "shortcut connections", which allow the gradient to be directly backpropagated to earlier layers. The main benefit of this is that it helps to mitigate the problem of vanishing gradient, which is a common issue with deep neural networks.

In a ResNet, the output of one layer is added to the output of a layer a few steps earlier before being passed through a ReLU activation function. This allows the network to learn so-called "residual functions" with reference to the layer inputs, which can help to train deeper models.

ResNet has been widely used in the field of deep learning, especially in tasks that involve image or video processing. It has achieved state-of-the-art performance in various benchmarks and competitions.
```

I replicated a model described in Section 4.2 of original paper.
However I failed to get the same classification error with just 64k iteration.
The author said terminate training at 64k iteration, however I got classification accuracy of 90.45% with almost 200 epochs.