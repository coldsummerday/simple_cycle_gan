import os.path
from dataset.base_dataset import BaseDataset, get_transform, get_params
from dataset.image_folder import make_dataset
from PIL import Image
import random


class UnalignedOriDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/dataset/trainA'
    and from domain B '/path/to/dataset/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/dataset'.
    Similarly, you need to prepare two directories:
    '/path/to/dataset/testA' and '/path/to/dataset/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/dataset/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/dataset/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/dataset/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/dataset/trainB'
        print ('loading domain A: %d'% len(self.A_paths))
        print ('loading domain B: %d'% len(self.B_paths))
        if opt.phase == 'test':
            if opt.end_index != -1:
                self.A_paths = self.A_paths[opt.start_index: opt.end_index]
                assert opt.serial_batches, 'cut images, but test with shuffle'
            else:
                self.A_paths = self.A_paths[opt.start_index: ]
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.is_test =  'test' in opt.phase

    def __getitem__(self, index):
        """Return a dataset point and its metadata information.

        Parameters:
            index (int)      -- a random integer for dataset indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if 'test' in self.opt.phase and not 'v1' in self.opt.model:
            A_img = Image.open(A_path).convert('RGB')
            transform=get_transform(self.opt, params=None, grayscale=(self.input_nc == 1))
            A = transform(A_img)
            return {'A': A, 'A_paths': A_path}
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        if 'v1' in self.opt.model and self.is_test:
            base = 4 if 'unet' not in self.opt.netG else int(self.opt.netG.split('_')[-1])
            A_img = scale_width(A_img, self.opt.load_size, method=Image.BICUBIC, base=base)
            B_img = scale_width(B_img, self.opt.load_size, method=Image.BICUBIC, base=base)
            if A_img.size[0] > B_img.size[0]:
                B_img = B_img.resize(A_img.size)
            else:
                B_img = B_img.crop((0, 0, A_img.size[0], A_img.size[1]))
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if 'test' in self.opt.phase:
            return self.A_size//self.batch_size
        return max(self.A_size, self.B_size)
