import torch 
import torch.nn as nn
import torch.nn.functional as F
from base_modules import *

class Generator(nn.Module):
  def __init__(self, in_channels=3):
    super(Generator, self).__init__()
    # encoder stuff
    self.conv1 = ConvBlock(3,64)
    self.conv2 = ConvBlock(64,64)
    self.down_conv1 = DownConv(64,128)
    self.conv3 = ConvBlock(128,128)
    self.dsconv1 = DepthwiseSeparableConv(128,128,stride=1)
    self.down_conv2 = DownConv(128,256)
    self.conv4 = ConvBlock(256,256)

    # residual layers... Do we even need 8??
    # I'll use four for now...
    # irb: inverted residual block 
    self.irb1 = InverseResidualBlock(256)
    self.irb2 = InverseResidualBlock(256)
    self.irb3 = InverseResidualBlock(256)
    self.irb4 = InverseResidualBlock(256)
    
    # decoder stuff
    self.conv5 = ConvBlock(256,256)
    self.up_conv1 = UpConv(256, 128)
    self.dsconv2 = DepthwiseSeparableConv(128,128,stride=1)
    self.conv6 = ConvBlock(128,128)
    self.up_conv2 = UpConv(128, 64)
    self.conv7 = ConvBlock(64,64)
    self.conv8 = ConvBlock(64,64)
    self.final_conv_layer = nn.Conv2d(64,3,kernel_size=1,stride=1, padding=1)
    self.tanh_activation = nn.Tanh()
  
  def encode(self, x):
    """ @param x: x is [N x C x H x W] images
        @returns: I think its [N x 256 x H/4 x W/4]?
    """
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.down_conv1(x)
    x = self.conv3(x)
    x = self.dsconv1(x)
    x = self.down_conv2(x)
    x = self.conv4(x)
    return x

  def decode(self, x):
    """ @param x: x is [N x C x H x W] image
        @returns: I think its [N x 3 x 4*H x 4*W]?
    """
    x = self.conv5(x)
    x = self.up_conv1(x)
    x = self.dsconv2(x)
    x = self.conv6(x)
    x = self.up_conv2(x)
    x = self.conv7(x)
    x = self.conv8(x)
    x = self.final_conv_layer(x)
    x = self.tanh_activation(x)
    return x
    
  def residual_forward(self, x):
    """ a forward pass through the residual layers
        @param x: [N, 256, H, W] tensor
        @returns: [N, 256, H, W] tensor
    """
    x = self.irb1(x)
    x = self.irb2(x)
    x = self.irb3(x)
    x = self.irb4(x)
    return x
  
  def forward(self, x):
    """ @param x: [N x C x H x W] images
        @returns: [N x C x H x W] images
    """
    x = self.encode(x)
    x = self.residual_forward(x)
    x = self.decode(x)
    return x

  def save_model(self, path):
      torch.save(self.state_dict(), path)

  def load_model(self, path):
    self.load_state_dict(torch.load(path))
