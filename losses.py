import torch
import torch.nn as nn
import torchvision.models as models

def getVGGConv4_4():
    """ Returns: VGG19 model up until Conv4_4 Layer. 
    """
    vgg19 = models.vgg19(pretrained=True)
    VGG = nn.Sequential(*list(vgg19.features._modules.values())[:26])
    return VGG

class ContentLoss(nn.Module):
  """ content loss helps to make the generated image retain the 
      content of the input photo
  """

  def __init__(self, VGG):
    super(ContentLoss, self).__init__()
    self.VGG = VGG
    self.L1Loss = nn.L1Loss()
  
  def forward(self, generated, photo):
    generated = self.VGG(generated)
    photo = self.VGG(photo)
    return self.L1Loss(generated, photo)


class GrayscaleStyleLoss(nn.Module):
  """ grayscale style loss makes the generated images have
      the clear anime style on the texture and lines"
  """

  def __init__(self, VGG):
    super(GrayscaleStyleLoss, self).__init__()
    self.VGG = VGG
    self.L1Loss = nn.L1Loss()

  @staticmethod
  def gram_matrix(A):
    """ @param A: image [N x C x H x W]
        gram = A_unrolled @ A_unrolled.T
        @returns: gram matrix of A, of shape [N, C, C]
    """
    N,C,H,W = A.shape
    A_unrolled = A.reshape((N,C,H*W))
    A_unrolled_transpose = torch.transpose(A_unrolled, 1, 2)
    gram = torch.bmm(A_unrolled, A_unrolled_transpose)
    return gram

  def forward(self, generated, anime_gray):
    """ @param generated: images generated from generator, G(photo),
                          of shape [N x C x H x W]
        @param anime_gray: grayscale anime images, of shape
                           [N x C x H x W]
    """
    gram_generated = GrayscaleStyleLoss.gram_matrix(self.VGG(generated))
    gram_anime_gray = GrayscaleStyleLoss.gram_matrix(self.VGG(anime_gray))
    return self.L1Loss(gram_generated, gram_anime_gray) / generated.numel()


class ColorReconLoss(nn.Module):
  """ Loss used to combat the loss of color. Ensure generated images 
      have the color of the original photos.
  """
      
  def __init__(self):
    super(ColorReconLoss, self).__init__()
    self.L1Loss = nn.L1Loss()
    self.HuberLoss = nn.SmoothL1Loss()
  
  @staticmethod
  def rgb_to_ycbcr(input):
    """ @param input: [N x 3 x H x W]
        returns: YUV formatted version of the images 
        code is repurposed from here: 
        https://discuss.pytorch.org/t/how-to-change-a-batch-rgb-images-to-ycbcr-images-during-training/3799/2
        formula is from: https://en.wikipedia.org/wiki/YCbCr
    """
    output = torch.zeros(input.shape)
    output[:, 0, :, :] += input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :, :] * 24.966 + 16.
    output[:, 1, :, :] += input[:, 0, :, :] * -37.797 + input[:, 1, :, :] * 74.203 + input[:, 2, :, :] * 112. + 128.
    output[:, 2, :, :] += input[:, 0, :, :] * 112.0 + input[:, 1, :, :] * 93.786 + input[:, 2, :, :] * 18.214 + 128.
    return output

  @staticmethod
  def unstandardizeImage(images):
    """ @param images: [N x C x H x W] tensor representing standardized images
        @returns: [N x C x H x W] tensor representing RGB images (unclipped)

        Undoes ImageNet standardization
    """
    _mean=[0.485, 0.456, 0.406]; _std=[0.229, 0.224, 0.225]
    output = torch.zeros(images.shape)
    for i in range(3):
      output[:,i,:,:] += images[:,i,:,:] * _std[i]
      output[:,i,:,:] += images[:,i,:,:] + _mean[i]
    output *= 255.0
    return output
  
  def forward(self, generated, real_photos):
    """ @param generated: batch of generated anime images in RGB format,
                          of shape [N x 3 x H x W]
        @param real_photos: batch of real-life photos used to generate generated 
                            images, of shape [N x 3 x H x W]
    """
    # scale to rgb values
    generated = ColorReconLoss.unstandardizeImage(generated)
    real_photos = ColorReconLoss.unstandardizeImage(real_photos)
    
    # convert to YUV format
    generated_yuv = ColorReconLoss.rgb_to_ycbcr(generated)
    real_photos_yuv = ColorReconLoss.rgb_to_ycbcr(real_photos)
    
    # calculate loss for each channel
    y_loss = self.L1Loss(generated_yuv[:,0,:,:], real_photos_yuv[:,0,:,:])
    u_loss = self.HuberLoss(generated_yuv[:,1,:,:], real_photos_yuv[:,1,:,:])
    v_loss = self.HuberLoss(generated_yuv[:,2,:,:], real_photos_yuv[:,2,:,:])
    
    return (y_loss + u_loss + v_loss) / generated.numel()