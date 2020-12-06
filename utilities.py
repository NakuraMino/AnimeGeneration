import pickle
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt

def plot_images(images, title, fig_index):  
    scene_len, H, W, C = images.shape
    f = plt.figure(fig_index, figsize=(scene_len*4,4))
    for k in range(scene_len):
        plt.subplot(1,scene_len,k+1)
        img = images[k].astype(np.uint8)
        plt.imshow(img)
        plt.title(f"{title} {k+1}")
    fig_index += 1
    return fig_index


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
            
    return np_tensor


def readListFromPickle(file_name):
  open_file = open(file_name, "rb")
  loaded_list = pickle.load(open_file)
  return loaded_list


def saveListToPickle(path, loss_list):
  with open(path, 'wb') as fp:
    pickle.dump(loss_list, fp)


def listToAvg(a_list, interval=100):
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