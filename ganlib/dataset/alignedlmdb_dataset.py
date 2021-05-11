import os
from .base_dataset import BaseDataset, get_params, get_transform
from .base_dataset import __pad as padding
import torchvision.transforms as transforms
from PIL import Image
import lmdb
import six

class AlignedLMDBDataset(BaseDataset):
    """A dataset class for paired image dataset.

    """
    def __init__(self,opt):
        super(AlignedLMDBDataset, self).__init__(opt)


        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.source_lmdb_path = opt.source_lmdb
        self.targe_lmdb_path = opt.target_lmdb

        self.filtered_index_list = []
        self.num_samples = 0
        with lmdb.open(self.source_lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False).begin(
                write=False) as txn:
            self.num_samples = int(txn.get('num-samples'.encode()))
            self.filtered_index_list = [index + 1 for index in range(self.num_samples)]

    def __len__(self):
        return len(self.filtered_index_list)

    def open_lmdb(self):
        self.source_env = lmdb.open(self.source_lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        self.target_env = lmdb.open(self.targe_lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)


    def __getitem__(self, index):
        """Return a dataset point and its metadata information.
        Parameters:
            index (int)      -- a random integer for dataset indexing
        """
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]
        if not hasattr(self, 'source_env') or not hasattr(self,"target_env"):
            self.open_lmdb()

        with self.source_env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                source_img = Image.open(buf).convert('RGB')  # for color image
            except IOError:
                # make dummy image and dummy label for corrupted image.
                source_img = Image.new('RGB', (self.opt.crop_size[0], self.opt.crop_size[1]))
        with self.target_env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                target_img = Image.open(buf).convert('RGB')  # for color image
            except IOError:
                # make dummy image and dummy label for corrupted image.
                target_img = Image.new('RGB', (self.opt.crop_size[0], self.opt.crop_size[1]))

        source_img, target_img = padding(source_img, target_img)
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, self.opt.crop_size)
        A_transform_list = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #transform_params = get_params(self.opt, B.size)
        B_transform_list = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A_transform = transforms.Compose(A_transform_list)
        B_transform = transforms.Compose(B_transform_list)

        source_img = A_transform(source_img)
        target_img = B_transform(target_img)
        return {'A': source_img, 'B': target_img}