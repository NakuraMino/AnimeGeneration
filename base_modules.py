import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
  """ Convolution Block made of convolution, instance norm, 
  and lrelu activation
  """

  def __init__(self, in_channels, out_channels, kernel=3, stride=1):
    super(ConvBlock, self).__init__()
    padding = kernel // 2
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
    self.inst_norm = nn.InstanceNorm2d(out_channels)
    self.lrelu = nn.LeakyReLU(0.2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.inst_norm(x)
    x = self.lrelu(x)
    return x


class DepthwiseSeparableConv(nn.Module):
  """ depthwise separable convolution layer according to the following posts  
  Honestly not too sure if I implemented the depth-wise convolution correctly
  or not. Let's cross our fingers and hope I did.
  https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
  https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843
  """  

  def __init__(self, in_channels, out_channels, multiplier=4, kernel=3, stride=2):
    """ @spec.requires: in_channels % groups == 0
        *this is accounted for by code: in_channels*multiplier. No need be careful
    """
    super(DepthwiseSeparableConv, self).__init__()
    self.depthwise = nn.Conv2d(in_channels, in_channels*multiplier, kernel_size=kernel, stride=stride, padding=1, groups=in_channels)
    self.pointwise = nn.Conv2d(in_channels*multiplier, out_channels, kernel_size=1, stride=1)    
    self.inst_norm = nn.InstanceNorm2d(out_channels)
    self.lrelu = nn.LeakyReLU(0.2)

  def forward(self, x):
    """ @param x: [N x C x H x W]
        @returns: [N x C x H/stride x W/stride] tensor 
        Note: Assuming stride is either 1 or 2
    """
    x = self.depthwise(x)
    x = self.pointwise(x)
    x = self.inst_norm(x)
    x = self.lrelu(x)
    return x


class DSConv(nn.Module):
    
  def __init__(self, in_channels, out_channels, kernel=3, stride=2):
    super(DSConv, self).__init__()
    self.depthwise = DepthwiseSeparableConv(in_channels, out_channels,stride=stride)
    self.conv_block1 = ConvBlock(out_channels, out_channels, kernel=1, stride=1)

  def forward(self, x):
    """ @param x: [N x C x H x W]
        @returns: [N x C x H/stride x W/stride] tensor 
        Note: Assuming stride is either 1 or 2
    """
    x = self.depthwise(x)
    # x is [N x out_channels x H/2 x W/2] if stride=2
    x = self.conv_block1(x)
    return x

class InverseResidualBlock(nn.Module):
  """ an inverse residual block :D Let's hope it's right...
  """

  def __init__(self, in_channels, middle_channels=512):
    super(InverseResidualBlock, self).__init__()
    self.conv_block = ConvBlock(in_channels, middle_channels, kernel=1,stride=1)
    self.dconv = DepthwiseSeparableConv(middle_channels, middle_channels//2, kernel=3, stride=1)
    self.conv = nn.Conv2d(middle_channels//2, in_channels, kernel_size=1, stride=1)
    self.inst_norm = nn.InstanceNorm2d(in_channels)

  def forward(self, x):
    """ @param x: [N x C x H x W]
        @returns: [N x C x H x W] tensor
    """
    residual = x
    x = self.conv_block(x)
    x = self.dconv(x)
    x = self.conv(x)
    x = self.inst_norm(x)
    return x + residual


class DownConv(nn.Module):
  """ Downsizes an input by a factor of two
  """

  def __init__(self, in_channels, out_channels):
    super(DownConv, self).__init__()
    self.dconv1 = DSConv(in_channels, out_channels, kernel=3, stride=2)
    self.dconv2 = DSConv(in_channels, out_channels, kernel=3, stride=1)

  def forward(self, x):
    """ @param x: [N x C x H x W]
        @returns x: [N x C x H/2 x W/2]
    """
    residual = x
    residual = self.dconv1(x)
    # residual: [N x out_channels x H/2 x W/2]

    x = F.interpolate(x, scale_factor=0.5)
    x = self.dconv2(x)
    # x: [N x out_channels x H/2 x W/2]
    
    return x + residual

class UpConv(nn.Module):
  """ Upsamples an input by a factor of two
  """

  def __init__(self, in_channels, out_channels):
    super(UpConv, self).__init__()
    self.dconv1 = DSConv(in_channels, out_channels, kernel=3,stride=1)

  def forward(self, x):
    """ @param x: [N x C x H x W]
        @returns x: [N x C x 2*H x 2*W]
    """
    N, C, H, W = x.shape
    x = F.interpolate(x, scale_factor=2)
    x = self.dconv1(x)
    return x