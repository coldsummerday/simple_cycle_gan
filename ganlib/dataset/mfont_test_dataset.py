import os.path
from dataset.base_dataset import BaseDataset, get_params, get_transform
from dataset.base_dataset import __pad as padding
from dataset.base_dataset import __scale_width as scale_width
from dataset.image_folder import make_dataset
from PIL import Image
import torch
import random

def cut_parse(paths):
    query = {}
    for path in paths:
        key = path.split('/')[-1].split('_')[0]
        if key not in query:
            query[key] = []
        query[key].append(path)
    return query

class MFontTestDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/dataset/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/dataset/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase+'A')  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.dir_B = os.path.join(opt.dataroot, opt.phase+'B')
        B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.B_query = cut_parse(B_paths)
        print ('search keys: %s' % str(self.B_query.keys()))
        if opt.phase == 'test':
            if opt.end_index != -1:
                self.AB_paths = self.AB_paths[opt.start_index: opt.end_index]
                assert opt.serial_batches, 'cut images, but test with shuffle'
            else:
                self.AB_paths = self.AB_paths[opt.start_index: ]
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.is_test = 'test' in opt.phase

    def __getitem__(self, index):
        """Return a dataset point and its metadata information.

        Parameters:
            index - - a random integer for dataset indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        A = Image.open(AB_path).convert('RGB')
        A = scale_width(A, target_width=self.opt.load_size)
            
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

        A_tensor = A_transform(A)
        #A_transform = get_transform(self.opt, transform_params, grayscale=True)
        for i in range(20):
            #key = AB_path.split('/')[-1].split('_')[0]
            key = '住址'
            path = self.B_query[key][random.randint(0, len(self.B_query[key])-1)]
            other = Image.open(path).convert('RGB')
            other = scale_width(other, target_width=self.opt.load_size)
            _, other = padding(A, other)
            other = A_transform(other)
            A_tensor = torch.cat((A_tensor, other), 0)
        return {'A': A_tensor, 'A_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
