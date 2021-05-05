"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        #if new_h == opt.load_size:
        if min(new_h, new_w) == opt.load_size:
            pass
        elif new_w <= new_h:
            new_w = opt.load_size
            new_h = opt.load_size * h // w
        else:
            new_h = opt.load_size
            new_w = opt.load_size * w // h
    base = 4 if 'unet' not in opt.netG else int(opt.netG.split('_')[-1])
    new_h = int(round(new_h / base) * base)
    new_w = int(round(new_w / base) * base)
    if isinstance(opt.crop_size, list):
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size[1]))
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size[0]))
    else:
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    ort = random.choice([0,1])
    return {'crop_pos': (x, y), 'flip': flip, 'ort': ort}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        base = 4 if 'unet' not in opt.netG else int(opt.netG.split('_')[-1])
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method, base)))
    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'], params['ort'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC, base=4):
    ow, oh = img.size
    if (min(oh, ow) == target_width):
        w, h = ow, oh
    elif ow <= oh:
        w = target_width
        h = int(target_width * oh / ow)
    else:
        h = target_width
        w = int(target_width * ow / oh)
    h = int(round(h / base) * base)
    w = int(round(w / base) * base)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    if len(size) == 2:
        th, tw = size
    else:
        tw = th = size

    if (ow >= tw and oh >= th):
        #print ('ow >= tw and oh >= th', img.size, (tw,th))
        return img.crop((x1, y1, x1 + tw, y1 + th))
    if ow >= oh:
        target = Image.new('RGB', (tw,th))
        num_copy = tw//ow + int(tw%ow!=0)
        left = 0
        for num in range(num_copy):
            right = min(ow*(num+1), tw)
            target.paste(img.crop((0, 0, right-left, oh)), (left, 0, right, oh))
            left = right
        #target.save('test_target.png')
        #print ('ow>=oh', img.size, target.size)
    else:
        target = Image.new('RGB', (tw,th))
        num_copy = th//oh + int(th%oh==0)
        up = 0
        for num in range(num_copy):
            bottom = min(oh*(num+1), th)
            target.paste(img.crop((0, 0, ow, bottom-up)), (0, up, ow, bottom))
            up = bottom
    return target

def __pad(A, B):
    wa, ha = A.size
    wb, hb = B.size
    assert ha == hb
    if wa <= wb:
        return A, B.crop((0,0,wa, ha))
    else:
        target = Image.new('RGB', (wa,ha))
        num_copy = wa//wb + int(wa%wb!=0)
        left = 0
        for num in range(num_copy):
            right = min(wb*(num+1), wa)
            target.paste(B.crop((0, 0, right-left, ha)), (left, 0, right, ha))
            left = right
        return A, target

def __flip(img, flip, ort):
    if flip:
        if ort == 0:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
