import os.path
from dataset.base_dataset import BaseDataset, get_transform, get_params
from dataset.base_dataset import __scale_width as scale_width
from dataset.image_folder import make_dataset
from dataset.base_dataset import __pad as padding
from PIL import Image
import random
import torch
import numpy as np

class ShuffleDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/dataset/trainA'
    and from domain B '/path/to/dataset/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/dataset'.
    Similarly, you need to prepare two directories:
    '/path/to/dataset/testA' and '/path/to/dataset/testB' during test time.
    """

    def __init__(self, opt, data_dir, data_list):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.max_length = 60
        self.opt = opt
        self.input_nc = self.opt.output_nc
        self.data_dir = data_dir
        
        self.labels = []
        self.paths = []
        self.label_lens = []
        with open(data_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                # label index
                if len(line.split(' ')) != 2: 
                    continue
                codes = line.split(' ')[1].split(',')
                if len(codes) > self.max_length:
                    continue
                if codes[-1] == '':
                    codes.remove('')
                img_code = [int(code)+1 for code in codes if int(code) < 6097]
                self.label_lens.append(len(img_code))
                # 把标签索引改为等长，后面填充0
                length = len(img_code)
                if length < self.max_length:
                    img_code += [0] * (self.max_length - length)
                self.labels.append(img_code)
                self.paths.append(line.split(' ')[0])

        print ('loading from {:s} : {:d}'.format(data_list, len(self.paths)))
        self.size = len(self.paths)  

    def __getitem__(self, index):
        """Return a dataset point and its metadata information.
        Parameters:
            index (int)      -- a random integer for dataset indexing
        """
        path = self.paths[index]  # make sure index is within then range
        img = Image.open(os.path.join(self.data_dir, path)).convert('RGB') 
        width = img.size[0]
        # apply image transformation
        params = get_params(self.opt, img.size)
        transform=get_transform(self.opt, params=params, grayscale=(self.input_nc == 1))
        im = transform(img)
        
        label = torch.from_numpy(np.array(self.labels[index]))
        label_len = torch.from_numpy(np.array(self.label_lens[index]))

        return im, label, label_len, width

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size 
  
def concat_dataset(s_trainset, t_trainset):
    class catDataset(Dataset):
        def __init__(self, s_trainset, t_trainset):
            self.ds1 = s_trainset
            self.ds2 = t_trainset
        def __len__(self):
            return max(len(self.ds1), len(self.ds2))
        def __getitem__(self, idx):
            sample = {
                "A" : self.ds1[idx % len(self.ds1)][0],
                "A_labels" : self.ds1[idx % len(self.ds1)][1],
                "A_label_lens" : self.ds1[idx % len(self.ds1)][2],
                "A_width" : self.ds1[idx % len(self.ds1)][3],
                "B" : self.ds2[idx % len(self.ds2)][0],
                "B_labels" : self.ds2[idx % len(self.ds2)][1],
                "B_label_lens" : self.ds2[idx % len(self.ds2)][2],
                "B_width" : self.ds2[idx % len(self.ds2)][3]
            } 
            return sample
    return catDataset(s_trainset, t_trainset)