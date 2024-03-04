# Import 
import torch
# import PASCAL VOC Dataset
from torchvision.datasets import VOCDetection
# import FasterRCNN model
from model import FasterRCNN

# define anchors for Pascal VOC dataset
# For anchors, we use 3 scales with box areas of 128^2, 256^2, and 512^2 pixels, 
# and 3 aspect ratios of 1:1, 1:2, and 2:1
anchors = [
    (188, 111), (113, 114), (70, 92),
    (416, 229), (261, 284), (174, 332),
    (768, 437), (499, 501), (355, 715)
]
# ignore cross-boundary anchors to not to contribute to the loss

# To reduce redundancy, non-maximum suppression (NMS) is used to merge highly overlapped anchors.
# We use a threshold of 0.7 to merge anchors.


# For training RPNs, we assign a binary class label to each anchor.
# we assign a positive label to two kinds of anchors
# (i) the anchor/anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box, or
# (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box. Note that a single ground-truth box may assign positive labels to multiple anchors.

# We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes. 
# Anchors that are neither positive nor negative do not contribute to the training objective.

# Our loss function for an image is defined as:
# L({pi}, {ti}) = (1/Ncls) * sum_i Lcls(pi, pi*) + lambda/Nreg * sum_i pi* * Lreg(ti, ti*)
# where
# pi is the predicted probability of anchor i being an object
# pi* is the ground-truth label of anchor i
# ti is the predicted 4-parameter vector for anchor i
# ti* is the ground-truth 4-parameter vector for anchor i
# Lcls is the log loss function (binary cross entropy)
# Lreg is the smooth L1 loss function
# Ncls is the number of anchor locations that predict an object
# Nreg is the number of anchor locations that predict an object
# lambda is a balancing parameter
#

# Lreg(ti, ti*) = sum_i sum_j pi* smoothL1(ti - ti*)
# Lcls(pi, pi*) = sum_i log(1 + exp(-pi)) - pi * pi*

# Optimization

# Instead, we randomly
# sample 256 anchors in an image to compute the loss function of a mini-batch, where the sampled
# positive and negative anchors have a ratio of up to 1:1. If there are fewer than 128 positive samples
# in an image, we pad the mini-batch with negative ones

# We use stochastic gradient descent (SGD) with a learning rate of 0.001, a momentum of 0.9, and a weight decay of 0.0005.
# We train for 60k iterations with a batch size of 2 images on 8 GPUs.
# We reduce the learning rate by a factor of 10 after 40k and 50k iterations.
# We use a warm-up learning rate of 0.0001 for the first 5k iterations.
# We use a weight decay of 0.0001 and a momentum of 0.9.
# We use a learning rate of 0.01 for the first 20k iterations and 0.001 for the next 10k iterations.



