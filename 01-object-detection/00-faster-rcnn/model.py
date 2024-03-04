"""
    Faster R-CNN Implementation
    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (https://arxiv.org/abs/1506.01497)

    The original Faster R-CNN code is revisited to understand the architecture
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py

"""
import torch
import torch.nn as nn
from torchvision.models import resnet50

class RoiPooling(nn.Module):
    """
    Region of Interest Pooling
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class RPN(nn.Module):
    """
    Region Proposal Network
    input: an image (of any size)
    output: a set of rectangular object proposals with an objectness score

    """
    def __init__(self, in_channel, intermediate_channel, n=3, k=9):
        """
        n: kernel size
        k: number of anchors
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channel, intermediate_channel, kernel_size=n, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(intermediate_channel)
        # This architecture is natuarally implemented with a n x n sliding window, 
        # followed by two sibling 1 x 1 conv layers(reg, cls respectively)
        self.cls = nn.Conv2d(intermediate_channel, 2 * k, kernel_size=1, stride=1, padding=0)
        self.reg = nn.Conv2d(intermediate_channel, 4 * k, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        cls_score = self.cls(x)
        bbox_pred = self.reg(x)
        return cls_score, bbox_pred

class FasterRCNN(nn.Module):
    """
    Faster R-CNN Prediction Network
    """
    def __init__(self):
        super(FasterRCNN, self).__init__()
        # use ResNet50 as a backbone
        self.backbone = resnet50()
        # Region Proposal Network
        self.rpn = RPN()
        # RoI Pooling
        self.roi_pooling = RoiPooling()
        
        self._initialize_weights()
    def forward(self, x):
        # Backbone
        x = self.backbone(x)
        # Region Proposal Network
        rpn = self.rpn(x)
        # RoI Pooling
        roi = self.roi_pooling(x, rpn)
        # Fully Connected Layer
        fc = self.fc(roi)
        # Classification Layer
        cls = self.cls(fc)
        # Regression Layer
        reg = self.reg(fc)

        # calculate loss

        return cls, reg
    def _initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.bias, 0, 0.01)