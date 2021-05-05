import os.path
from .base_dataset import BaseDataset, get_params, get_transform
from .base_dataset import __pad as padding
from .image_folder import make_dataset
from PIL import Image


class FontDataset(BaseDataset):
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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        #assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
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
        AB = Image.open(AB_path).convert('RGB')

        # split AB image into A and B
        w, h = AB.size
        h2 = int(h/2)
        h -= int(h%2!=0)
        B = AB.crop((0, 0, w, h2))
        A = AB.crop((0, h2, w, h))
        if self.is_test:
            A, B = padding(A, B)
        #if 'v1' in self.opt.model:
        #    base = 4 if 'unet' not in self.opt.netG else int(self.opt.netG.split('_')[-1])
        #    A = scale_width(A, self.opt.load_size, method=Image.BICUBIC, base=base)
        #    B = scale_width(B, self.opt.load_size, method=Image.BICUBIC, base=base)
        #    if A.size[0] > B.size[0]:
        #        B = B.resize(A.size)
        #    else:
        #        B = B.crop((0, 0, A.size[0], A.size[1]))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
