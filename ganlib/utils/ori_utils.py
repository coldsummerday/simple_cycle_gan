"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict



def save_checkpoint(model, filename, num_epoch:int=0,num_iter:int=0):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain some train info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    os.makedirs(os.path.dirname(filename),exist_ok=True)

    if hasattr(model, 'module'):
        model = model.module

    checkpoint = {
        'num_epoch': num_epoch,
        'num_iter':num_iter,
        'state_dict': weights_to_cpu(model.state_dict())
    }
    torch.save(checkpoint, filename)

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on CPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def load_checkpoint(model,filename:str,device):
    checkpoint = torch.load(filename, map_location=device)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        # load state_dict
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict,strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    return checkpoint


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


class catDataset(Dataset):
    def __init__(self, s_trainset, t_trainset):
        super(catDataset, self).__init__()
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

def concat_dataset(s_trainset, t_trainset):
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