import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
import cv2
import random 

""" AUXILIARY FUNCTIONS
    A lot of credit goes to Chris Xie for his code on dataloader stuff.
"""

def standardize_images(image):
    """ Convert a numpy.ndarray [N x H x W x 3] of images to [0,1] range, and then standardizes
        @return: a [N x H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]
    return image_standardized

def image_to_tensor(image):
    if image.ndim == 4: # NHWC
        tensor = torch.from_numpy(image).permute(0,3,1,2).float()
    elif image.ndim == 3: # HWC
        tensor = torch.from_numpy(image).permute(2,0,1).float()
    return tensor

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

""" DATASETS AND DATALOADERS
"""

class PhotoAndAnimeDataset(Dataset): 
    """ dataloader for photos, original anime, grayscale smooth anime, and smoothed anime images
        where labels are photos=0, original=1, smoothed=0, photos=0
    """
    def __init__(self, anime_dir, photo_dir): 
        self.anime_dir = anime_dir
        self.photo_dir = photo_dir

        self.anime_images = os.listdir(anime_dir)
        self.photo_images = os.listdir(photo_dir)

    def __len__(self): 
        return max(len(self.anime_images), len(self.photo_images))

    def __getitem__(self, idx):
        # retrieve data for Generator training 
        idx = random.randrange(0, len(self.photo_images))
        im_path = self.photo_images[idx]
        photo = imageio.imread(im_path, pilmode='RGB')

        idx = random.randrange(0, len(self.anime_images))
        im_path = self.anime_images[idx]
        anime = imageio.imread(im_path, pilmode='RGB')

        # retrieve data for discriminator training
        is_anime = random.randint(0,4)
        label = None; image = None
        if is_anime != 3: 
            idx = random.randrange(0, len(self.anime_images))
            im_path = self.anime_images[idx]
            smooth_gray_original = random.randrange(0, 3)
            if smooth_gray_original == 0: 
                # smooth
                image = imageio.imread(im_path, pilmode='RGB')
                image = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
                label = 0
            elif smooth_gray_original == 1:
                # grayscale smooth
                image = imageio.imread(im_path, pilmode='L')
                image = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
                label = 0
            else:
                image = imageio.imread(im_path, pilmode='RGB')
                label = 1
        else: 
            idx = random.randrange(0, len(self.anime_images))
            im_path = self.photo_images[idx]
            image = imageio.imread(im_path, pilmode='RGB')
            label = 0

        photo = standardize_images(photo)
        photo = image_to_tensor(photo)
        
        image = standardize_images(image)
        image = image_to_tensor(image)

        anime = standardize_images(anime)
        anime = image_to_tensor(anime)
        return {'anime': anime, 'photo': photo, 'image': image, 'label': label}

def getPhotoAndAnimeDataloader(anime_dir, photo_dir, batch_size=4, num_workers=4, shuffle=True):
    dataset = PhotoAndAnimeDataset(anime_dir, photo_dir)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)