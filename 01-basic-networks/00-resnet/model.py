"""
 ResNet Implementation
 Deep Residual Leanring for Image Recognition (https://arxiv.org/abs/1512.03385)

 The original ResNet code is revisited to understand the architecture
 https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

 I referred to the following site to understand the architecture
 https://pseudo-lab.github.io/pytorch-guide/docs/ch03-1.html
"""
import torch
import torch.nn as nn

# 3x3 convolution
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )
    
# A Building Block shown in Figure 2.
class BasicBlock(nn.Module):
    """
    BasicBlock for ResNet
    """
    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # Page 4 in paper
        # Batch normalization right after each convolution and before activation 
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # The parameter-free identify shortcuts are paricularly important for the bottlenet architectures.
        # If the identify shortcut in Fig. 5 (right) is replaced with projection,
        # one can show that the time complexity and model size are doubled, 
        # as the shortcut is connected to the two high-dimensional ends.
        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            # if input and output are different
            residual = self.shortcut(x)

        out += residual #identity function
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet model for 4.2. CIFAR-10 dataset and analysis page 7 in the paper:
    input: The network inputs are 32 x 32 images, with the per-pixel mean subtracted.
    output: probability of the class
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        # The first layer is 3x3 convolutions.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # Then we use a stack of 6n layers with 3x3 convolutions on the feature maps of sizes 
        # {32, 16, 8} respectively, with 2n(num_blocks) layers for each feature map size.
        # The numbers of filters are {16, 32, 64} respectively.
        # The subsampling is performed by convolutions with a stride of 2.
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # The network ends with a global average pooling, a 10-way fully-connected layer, and softmax.
        # There are totally 6n+2 stacked weighted layers.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self._initialize_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            # Page 3 in paper:
            # The dimensions of x and F must be equal in Eqn.(1).
            # If this is not the case, (e.g. when changing the input/output channels),
            # we can perform a linear projection Ws by the shortcut connections to match the dimensions.
            # then W_s becomes trainable
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
            Weight initialization technique in 
            Delving deep into rectifiers:Surpassing human-level performance on imagenet classification
            (https://arxiv.org/abs/1502.01852) as the original paper mentioned

            zero-mean gaussian distribution whose standard deviation is sqrt(2/n_l)
            where n_l is the number of connections of a response
        """
        # For the first layer(l = 1), we should have n_1 Var[w_1] = 1
        # because there is not ReLU applied on the input signal, but factor doesn't matter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming He initialization weights ~ N(0, std^2), std = gain/sqrt(fan_mode)
                # gain is automatically set to 2 in "relu" option
                # fan_mode is "fan_out" to use number of connections of a response
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # Others bias = 0
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x