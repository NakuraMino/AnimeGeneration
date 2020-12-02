import torch 
import torch.nn as nn

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