import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio

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

class PhotoDataset(Dataset):
    """ dataloader for photo images only
        no labels needed.
    """

    def __init__(self, base_dir, grayscale=False):
        self.base_dir = base_dir
        self.all_images = os.listdir(base_dir)
        self.len = len(self.all_images)
        self.grayscale = "L" if grayscale else "RGB"

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_path = self.base_dir + self.all_images[idx]
        image = imageio.imread(image_path, pilmode=self.grayscale)
        if self.grayscale == 0:
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            image = np.tile(image, (1,1,3))
        image = standardize_images(image)
        image = image_to_tensor(image)
        return image
        
def getPhotoDataloader(base_dir, batch_size=4, num_workers=4, shuffle=True):
    dataset = PhotoDataset(base_dir)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)

class PhotoAndAnimeDataset(Dataset):
    """ dataloader for photos, original anime, and smoothed anime images
        where labels are original=1, smoothed=0, photos=0
    """

    def __init__(self, anime_original_dir, anime_smooth_dir, photo_base_dir, ratio=4):
        # first fourth is original anime images
        self.anime_original_dir = anime_original_dir
        self.anime_images = os.listdir(anime_original_dir)
        self.num_anime_photos = len(self.anime_images) * 2 * ratio

        # second fourth is smoothed gray anime images
        self.anime_smooth_dir = anime_smooth_dir
        self.anime_smooth_images = os.listdir(anime_smooth_dir)
        self.num_smooth_images = len(self.anime_smooth_images) * (ratio - 1)
        # print(self.num_smooth_images)

        # third fourth is gray anime images
        self.anime_gray_dir = anime_original_dir
        self.anime_gray_images = self.anime_images.copy()
        self.num_anime_gray_images = len(self.anime_images) 

        # final third is legit photos
        self.photo_base_dir = photo_base_dir
        self.photo_images = os.listdir(photo_base_dir)

        # append all together
        self.all_images = self.anime_images.copy()
        
        for i in range((2 * ratio) - 1):
            self.all_images.extend(self.anime_images)
        
        for i in range(ratio):
            self.all_images.extend(self.anime_smooth_images)
        
        self.all_images.extend(self.photo_images)
        self.len = len(self.all_images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """ 1s for anime photos and 0s for real photos and smoothed photos
        """
        # grab image path and label
        label = None
        image = None
        if idx < self.num_anime_photos:
            image_path = self.anime_original_dir + self.all_images[idx]
            image = imageio.imread(image_path, pilmode='RGB') # colored
            label = torch.ones((1,64,64))
        elif idx < self.num_anime_photos + self.num_smooth_images:
            image_path = self.anime_smooth_dir + self.all_images[idx]
            label = torch.zeros((1,64,64))
            image = imageio.imread(image_path, pilmode='L') # gray
        elif idx < self.num_anime_photos + self.num_smooth_images + self.num_anime_gray_images:
            image_path = self.anime_original_dir + self.all_images[idx]
            label = torch.zeros((1,64,64))
            image = imageio.imread(image_path, pilmode='L') # gray    
        else:
            image_path = self.photo_base_dir + self.all_images[idx]
            label = torch.zeros((1,64,64))
            image = imageio.imread(image_path, pilmode='RGB') # colored
        
        # make 3 layers if its grayscaled
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.tile(image, (1,1,3))

        image = standardize_images(image)
        image = image_to_tensor(image)
        
        return image, label
        
def getPhotoAndAnimeDataloader(anime_base_dir, smooth_dir, photo_base_dir, batch_size=4, num_workers=4, shuffle=True):
    dataset = PhotoAndAnimeDataset(anime_base_dir, smooth_dir, photo_base_dir)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)

class AnimeDataset(Dataset):
    """ dataloader for anime images
        Am I dumb? Is this not needed at all??
        No labels needed.
    """

    def __init__(self, base_dir, grayscale=False):
        self.base_dir = base_dir
        self.all_images = os.listdir(base_dir)
        self.len = len(self.all_images)
        self.grayscale = "L" if grayscale else "RGB"

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_path = self.base_dir + self.all_images[idx]
        # print(image_path)
        image = imageio.imread(image_path, pilmode=self.grayscale)
        if self.grayscale == 0:
          image = np.stack([image,image,image], axis=-1)
        image = standardize_images(image)
        image = image_to_tensor(image)
        return image
        
def getAnimeDataloader(base_dir, batch_size=4, grayscale=False, num_workers=4, shuffle=True):
    dataset = AnimeDataset(base_dir, grayscale=grayscale)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)