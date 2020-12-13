import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Conv2d(in_channel, out_channel, kernel_size)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(3, 32, 3, stride=1, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=1, padding=1))
        self.norm1 = nn.InstanceNorm2d(128)
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, stride=2, padding=1))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=1))
        self.norm2 = nn.InstanceNorm2d(256)
        self.conv6 = nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, stride=1, padding=1))
        self.norm3 = nn.InstanceNorm2d(256)
        self.conv7 = nn.utils.spectral_norm(nn.Conv2d(256, 1, 3, stride=1, padding=1))

        self.dropout = nn.Dropout2d(p=0.2)

            
    def forward(self, x, hidden_state=None):
        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv6(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv7(x)
        return x
    
    def save_model(self, path):
      torch.save(self.state_dict(), path)

    def load_model(self, path):
      self.load_state_dict(torch.load(path))
