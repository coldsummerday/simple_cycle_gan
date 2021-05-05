import os.path
from .base_dataset import BaseDataset, get_transform, get_params
from PIL import Image
import random
import torch, cv2
import numpy as np
import torchvision.transforms as transforms
from .data_augment import Transform

class ShuffleDataset(BaseDataset):
    def __init__(self, opt, data_list, augment=0, shuffle=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.max_length = 60
        self.opt = opt
        self.input_nc = self.opt.output_nc
        self.augment = augment
        self.data_list = data_list
        self.shuffle = shuffle

        self.labels = []
        self.paths = []
        self.label_lens = []
        self.build_dataset()  # build dataset
        self.size = len(self.paths)


    def build_dataset(self):
        with open(self.data_list, 'r') as f:
            lines = f.readlines()
            if self.shuffle:
                random.shuffle(lines)
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
                img_code += [0] * (self.max_length - len(img_code))
                self.labels.append(img_code)
                self.paths.append(line.split(' ')[0])

    def __getitem__(self, index):
        """Return a dataset point and its metadata information.
        Parameters:
            index (int)      -- a random integer for dataset indexing
        """
        path = self.paths[index]  # make sure index is within then range
        img = Image.open(path).convert('RGB')

        # 数据增强
        if self.augment:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            trans = Transform(img)
            img = trans.data_augment()
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        # apply image transformation
        params = get_params(self.opt, img.size)
        transform_list = get_transform(self.opt, params=params, grayscale=(self.input_nc == 1))
        img = transform_list[0](img)
        width = img.size[0]
        if width > 640:
            width = 640
        transform = transforms.Compose(transform_list[1:])
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

