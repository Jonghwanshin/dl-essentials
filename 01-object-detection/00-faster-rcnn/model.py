"""
    Faster R-CNN Implementation
    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (https://arxiv.org/abs/1506.01497)

    The original Faster R-CNN code is revisited to understand the architecture
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py

"""
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50

class RoiPooling(nn.Module):
    """
    Region of Interest Pooling uses max pooling to convert the features inside any valid region of interest
    into a small feature map with a fixed spatial extent of H x W (e.g. 7 x 7).

    where H & W are layer hyper-parameters that are independent of any particular ROI.

    In this paper, an RoI is a rectangular window into a conv feature map.
    Each RoI is defined by a four-tuple (r, c, h, w) that specifies its top-left corner (r, c) and its height and width (h, w).
    RoI max pooling works by dividing the h x w RoI window into H x W grid of sub-windows of approximate size
    h/H x w/W and then max-pooling the values in each sub window into the corresponding output grid cell.

    Pooling is applied independently to each feature map channel, as in standard max pooling.

    Reference
    - Source code for torchvison.ops.roi_pool: https://pytorch.org/vision/0.17/_modules/torchvision/ops/roi_pool.html
    - Fast-RCNN paper (https://arxiv.org/pdf/1504.08083.pdf)
    - "How does a pytorch function work?" - Stack overflow (https://stackoverflow.com/questions/73938616/how-does-a-pytorch-function-such-as-roipool-work)
    """
    def __init__(self, output_size, spatial_size: float = 1.0):
        super().__init__()
        self.output_size = output_size
        self.spatial_size = spatial_size

    def forward(self, input, boxes) -> Tensor:
        """
        takes the original feature map(input) and list of boxes(boxes)
        if output size for each boxes is described(output_size), it will return the pooled feature map for each box
        or it will return the pooled feature map for each box with scale of spatial_size
        Args:
            input: input tensor of shape (N, C, H, W)
            boxes: the box coordinates in (x1, y1, x2, y2) format in the input image
            output_size: the size of the output after the pooling
            spatial_scale: a scaling factor that maps the input coordinates to the box coordinates
        """
        output = nn.MaxPool2d(input, self.output_size, stride=1)
        return output

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
        # non maximum suppression
        self._initialize_weights()
    def forward(self, x):
        # Backbone
        x = self.backbone(x)
        # Region Proposal Network
        cls_score, bbox_pred = self.rpn(x)
        # RoI Pooling
        roi = self.roi_pooling(x, bbox_pred)
        # Fully Connected Layer
        fc = self.fc(roi)
        # Classification Layer
        cls = self.cls(fc)
        # Regression Layer
        reg = self.reg(fc)
        return cls, reg
    def _initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.bias, 0, 0.01)