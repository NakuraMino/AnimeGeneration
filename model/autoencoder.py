import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from base_modules import * 
import torchvision.models as models


class AutoEncoder(nn.Module):
  def __init__(self, in_channels=3):
    super(AutoEncoder, self).__init__()

    # encode using VGG19? 
    vgg19 = models.vgg19(pretrained=True)
    self.VGG = nn.Sequential(*list(vgg19.features._modules.values()))

    # residual layers 
    self.irb1 = InverseResidualBlock(512)
    self.irb2 = InverseResidualBlock(512)

    # decoder 
    self.conv1 = ConvBlock(512,512)
    self.up_conv1 = UpConv(512,256)

    self.conv2 = ConvBlock(256,256)
    self.up_conv2 = UpConv(256,128)

    self.conv3 = ConvBlock(128,128)
    self.up_conv3 = UpConv(128,64)

    self.conv4 = ConvBlock(64,64)
    self.up_conv4 = UpConv(64,32)

    self.conv5 = ConvBlock(32,32)
    self.up_conv5 = UpConv(32,16)

    self.final_conv_layer = nn.Conv2d(16,3,kernel_size=1,stride=1)
    

  def forward(self, x):
    """ @param x: [N x C x H x W] tensor
    """

    # assuming x is [4 x 3 x 256 x 256]
    x = self.VGG(x) 
    # x is [4 x 512 x 8 x 8]

    x = self.irb1(x)
    x = self.irb2(x)
    # x is [4 x 512 x 8 x 8]

    x = self.conv1(x)
    x = self.up_conv1(x)
    # x is [4 x 256 x 16 x 16]

    x = self.conv2(x)
    x = self.up_conv2(x)
    # x is [4 x 128 x 32 x 32]

    x = self.conv3(x)
    x = self.up_conv3(x)
    # x is [4 x 64 x 64 x 64]

    x = self.conv4(x)
    x = self.up_conv4(x)
    # x is [4 x 32 x 128 x 128]

    x = self.conv5(x)
    x = self.up_conv5(x)
    # x is [4 x 16 x 256 x 256]

    x = self.final_conv_layer(x)
    # x is [4 x 3 x 256 x 256]

    return x

  def save_model(self, path):
    torch.save(self.state_dict(), path)

  def load_model(self, path):
    self.load_state_dict(torch.load(path))