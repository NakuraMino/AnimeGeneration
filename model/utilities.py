import pickle
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL

def plot_images(images, title, fig_index):
    """ plot images in a row

        @param images: an [N x H x W x C] tensor 
        @fig_index: a value representing the figure index ex. fig_idx=1. Check
                    matplotlib for more information
    """  
    scene_len, H, W, C = images.shape
    f = plt.figure(fig_index, figsize=(scene_len*4,4))
    for k in range(scene_len):
        plt.subplot(1,scene_len,k+1)
        img = images[k].astype(np.uint8)
        plt.imshow(img)
        plt.title(f"{title} {k+1}")
    fig_index += 1
    return fig_index

def adjust_brightness_of_gen_images(images, gen_images):
    """ adjust the brightness of the generated images to match the brightness 
        of the original images. 

        @param images: [N x H x W x C] np.ndarray of original images 
        @param gen_images: [N x H x W x C] np.ndarray of generated images g(images) = gen_images
                           where g is the generator
        @returns: [N x H x W x C] np.ndarray of the generated images with adjusted brightness

        Note: constants to determine brightness are taken from 
              https://github.com/TachibanaYoshino/AnimeGAN/blob/master/tools/adjust_brightness.py
    """
    N, H, W, C = images.shape
    flat_images = np.reshape(images, (N,-1,C))
    flat_gen_images = np.reshape(gen_images, (N,-1,C))
    brightness = lambda im : 0.299 * im[:,:,0].mean(axis=1) + 0.587 * im[:,:,1].mean(axis=1) + 0.114 * im[:,:,2].mean(axis=1)
    images_brightness = brightness(images)
    gen_images_brightness = brightness(gen_images)
    brightness_diff = images_brightness / gen_images_brightness
    brightness_diff = brightness_diff.reshape((4,1,1,3))
    adj_image = np.clip(gen_images * brightness_diff, 0, 255)
    return adj_image 

def torch_to_numpy(torch_tensor, is_standardized_image = False):
    """ Converts torch tensor (...CHW) to numpy tensor (...HWC) for plotting
    
        If it's an rgb image, it puts it back in [0,255] range (and undoes ImageNet standardization)
    """
    np_tensor = torch_tensor.cpu().clone().detach().numpy()
    if np_tensor.ndim >= 4: # ...CHW -> ...HWC
        np_tensor = np.moveaxis(np_tensor, [-3,-2,-1], [-1,-3,-2])
    if is_standardized_image:
        _mean=[0.485, 0.456, 0.406]; _std=[0.229, 0.224, 0.225]
        for i in range(3):
            np_tensor[...,i] *= _std[i]
            np_tensor[...,i] += _mean[i]
        np_tensor *= 255
    np_tensor = np.clip(np_tensor, 0, 255)
    return np_tensor

def save_torch_as_images(file_path, torch_tensor, unique_identifier="", is_standardized_image=False, adjust_brightness=False, imgs=None):
    """ saves torch tensor as a sequence of images

        @param file_path: place to save file to
        @param torch_tensor: [N x C x H x W] tensor 
        @param is_standardized_image: whether the image is standardized or not
        @param adjust_brightness: where you want to adjust brightness or not
        @param imgs: the images to compare the brightness to. [N x C x H x W]
    """

    gen_images = torch_to_numpy(torch_tensor, is_standardized_image=is_standardized_image)
    if adjust_brightness and imgs is not None:
      imgs = torch_to_numpy(imgs, is_standardized_image=is_standardized_image)
      if imgs.shape == gen_images.shape:
        gen_images = adjust_brightness_of_gen_images(imgs, gen_images)

    scene_len, H, W, C = gen_images.shape

    for k in range(scene_len):
        img = gen_images[k].astype(np.uint8)
        img = Image.fromarray(img) 
        img = img.save(f"{file_path}{unique_identifier}{k}.jpg")

# import dataloader
# dataloader = dataloader.getPhotoDataloader('./large_photos/', num_workers=1)
# d_iter = iter(dataloader)
# original = next(d_iter)
# dark = original

# save_torch_as_images('./large_photos/', dark, unique_identifier="bleh", is_standardized_image=True, adjust_brightness=True, imgs=original)


def readListFromPickle(file_name):
  """ loads list from .pkl file

      @param file_name: path to .pkl file
      @returns: list 
  """
  try:
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    return loaded_list
  except:
      return []


def saveListToPickle(path, loss_list):
  """ Saves a list to a .pkl file

      @param path: path to save file
      @param loss_list: the list to save 
  """
  with open(path, 'wb') as fp:
    pickle.dump(loss_list, fp)


def listToAvg(a_list, interval=100):
  """ takes a list and averages the values at specified interval.
      
      @param a_list: the list to calculate averages from 
      @returns: shortened list
  """
  avg_list = []
  i = 0
  avg = 0.0
  for a in a_list:
    avg += a
    if i % interval == (interval - 1):
      avg_list.append(avg / 100)
      avg = 0
    i+=1
  return avg_list