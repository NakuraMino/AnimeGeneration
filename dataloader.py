import os 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 
from PIL import Image 

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


""" DATASETS AND DATALOADERS
"""

class PhotoDataset(Dataset):
    """ dataloader for photo images only
    """

    def __init__(self, base_dir, grayscale=False):
        self.base_dir = base_dir
        self.all_images = os.listdir(base_dir)
        self.len = len(self.all_images)
        self.grayscale = 0 if grayscale else 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_path = self.base_dir + self.all_images[idx]
        # print(image_path)
        image = cv2.imread(image_path, self.grayscale)
        image = standardize_images(image)
        image = image_to_tensor(image)
        return image
        
def getPhotoDataloader(base_dir, batch_size=4, shuffle=True):
    dataset = PhotoDataset(base_dir)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

# path = './dataset/train_photo/'
# ad = PhotoDataset(path)
# im = ad[1]
# print(im.shape)

# dl = getPhotoDataloader(path)
# batch = next(iter(dl))
# print(batch.shape)

class PhotoAndAnimeDataset(Dataset):
    """ dataloader for photo and anime images
    """

    def __init__(self, photo_base_dir, anime_base_dir, grayscale=False):
        # first half is anime images
        self.anime_base_dir = anime_base_dir
        self.anime_images = os.listdir(anime_base_dir)
        self.num_anime_photos = len(self.anime_images)

        # second half is legit images
        self.photo_base_dir = photo_base_dir
        self.photo_images = os.listdir(photo_base_dir)

        # append all together
        self.all_images = self.anime_images.copy()
        self.all_images.extend(self.photo_images)
        self.len = len(self.all_images)
        self.grayscale = 0 if grayscale else 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """ 1s for anime photos and 0s for real photos
            BUG: when using dataloader, the 1s and 0s are flipped, which 
            makes no sense since it works when using getitem directly.
            
            TEMPORARY FIX: just roll with the bug mann. I return the opposite
            values.

        """
        label = None
        if idx < self.num_anime_photos:
            image_path = self.anime_base_dir + self.all_images[idx]
            label = torch.zeros(1)
        else:
            image_path = self.photo_base_dir + self.all_images[idx]
            label = torch.ones(1)

        # print(image_path)
        image = cv2.imread(image_path, self.grayscale)
        image = standardize_images(image)
        image = image_to_tensor(image)
        
        return image, label
        
def getPhotoAndAnimeDataloader(photo_base_dir,anime_base_dir, batch_size=4, shuffle=True):
    dataset = PhotoAndAnimeDataset(photo_base_dir, anime_base_dir)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

# anime_path = './dataset/Shinkai/smooth/'
# photo_path = './dataset/train_photo/'

# paad = PhotoAndAnimeDataset(photo_path, anime_path)
# im, label = paad[2000]
# print(im.shape)
# print(label)

# dl = getPhotoAndAnimeDataloader(anime_path, photo_path, batch_size=4)
# images, labels = next(iter(dl))
# print(images.shape)
# print(labels)


class AnimeDataset(Dataset):
    """ dataloader for anime images
        Am I dumb? Is this not needed at all??
    """

    def __init__(self, base_dir, grayscale=False):
        self.base_dir = base_dir
        self.all_images = os.listdir(base_dir)
        self.len = len(self.all_images)
        self.grayscale = 0 if grayscale else 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_path = self.base_dir + self.all_images[idx]
        # print(image_path)
        image = cv2.imread(image_path, self.grayscale)
        image = standardize_images(image)
        image = image_to_tensor(image)
        return image
        
def AnimeDataloader(base_dir, batch_size=4, shuffle=True):
    dataset = AnimeDataset(base_dir)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

# path = './dataset/Shinkai/smooth/'
# ad = AnimeDataset(path)
# im = ad[1]
# print(im.shape)

# dl = getAnimeDataloader(path)
# batch = next(iter(dl))
# print(batch.shape)