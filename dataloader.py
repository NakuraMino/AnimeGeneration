import os 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 
from PIL import Image 

""" A lot of credit goes to Chris Xie for his code on dataloader stuff.
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

class AnimeDataset(Dataset):
    """ dataloader for anime images
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.all_images = os.listdir(base_dir)
        self.len = len(self.all_images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_path = self.base_dir + self.all_images[idx]
        # print(image_path)
        image = cv2.imread(image_path, 1)
        image = standardize_images(image)
        image = image_to_tensor(image)
        return image
        
def getThreeShapesDataloader(base_dir, batch_size=4, shuffle=True):
    dataset = AnimeDataset(base_dir)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

# path = './dataset/Shinkai/smooth/'
# ad = AnimeDataset(path)
# im = ad[1]
# print(im.shape)

# dl = getThreeShapesDataloader(path)
# batch = next(iter(dl))
# print(batch.shape)