"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from random import shuffle
from torch.utils.data import Dataset, DataLoader


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the dataset from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def concat_labels(label_idx, label_len):
    targets = []
    cnt = 0
    for i in range(label_idx.shape[0]):
        cnt += int(label_len[i])
        real_label_idx = label_idx[i][:int(label_len[i])]
        targets.extend(real_label_idx)
    targets = torch.from_numpy(np.array(targets))
    return targets


def concat_dataset(s_trainset, t_trainset):
    class catDataset(Dataset):
        def __init__(self, s_trainset, t_trainset):
            self.ds1 = s_trainset
            self.ds2 = t_trainset

        def __len__(self):
            return max(len(self.ds1), len(self.ds2))

        def __getitem__(self, idx):
            sample = {
                "S": self.ds1[idx % len(self.ds1)][0],
                "S_labels": self.ds1[idx % len(self.ds1)][1],
                "S_label_lens": self.ds1[idx % len(self.ds1)][2],
                "S_width": self.ds1[idx % len(self.ds1)][3],
                "T": self.ds2[idx % len(self.ds2)][0],
                "T_labels": self.ds2[idx % len(self.ds2)][1],
                "T_label_lens": self.ds2[idx % len(self.ds2)][2],
                "T_width": self.ds2[idx % len(self.ds2)][3]
            }
            return sample
    return catDataset(s_trainset, t_trainset)

def dict_data2model(model,data:dict):
    def tocuda(data):
        for key, values in data.items():
            if hasattr(values, 'cuda'):
                data[key] = values.cuda()
        return data
    if hasattr(model, "module"):
        if next(model.module.parameters()).is_cuda:
            data = tocuda(data)
    else:
        if next(model.parameters()).is_cuda:
            data = tocuda(data)
    return data