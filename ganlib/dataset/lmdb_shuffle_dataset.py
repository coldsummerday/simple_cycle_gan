import os.path
from .base_dataset import BaseDataset, get_transform, get_params
from PIL import Image
import torch, cv2
import numpy as np
import torchvision.transforms as transforms
from .data_augment import Transform
import lmdb
import six


class LMDBShuffleDataset(BaseDataset):
    def __init__(self, opt, lmdb_data_path:str,charset:str,augment=0, shuffle=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.max_length = 60
        self.opt = opt
        self.input_nc = self.opt.output_nc
        self.augment = augment
        self.shuffle = shuffle
        self.lmdb_data_path = lmdb_data_path

        self.charset = charset
        self.num_samples = 0
        with lmdb.open(lmdb_data_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
            self.num_samples = int(txn.get('num-samples'.encode()))
            self.filtered_index_list = [index + 1 for index in range(self.num_samples)]





    # def build_dataset(self):
    #     with open(self.data_list, 'r') as f:
    #         lines = f.readlines()
    #         if self.shuffle:
    #             random.shuffle(lines)
    #         for line in lines:
    #             line = line.strip()
    #             # label index
    #             if len(line.split(' ')) != 2:
    #                 continue
    #             codes = line.split(' ')[1].split(',')
    #             if len(codes) > self.max_length:
    #                 continue
    #             if codes[-1] == '':
    #                 codes.remove('')
    #             img_code = [int(code)+1 for code in codes if int(code) < 6097]
    #             self.label_lens.append(len(img_code))
    #             # 把标签索引改为等长，后面填充0
    #             img_code += [0] * (self.max_length - len(img_code))
    #             self.labels.append(img_code)
    #             self.paths.append(line.split(' ')[0])

    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_data_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        """Return a dataset point and its metadata information.
        Parameters:
            index (int)      -- a random integer for dataset indexing
        """
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]
        if not hasattr(self, 'env'):
            self.open_lmdb()
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:

                img = Image.open(buf).convert('RGB')  # for color image

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.

                img = Image.new('RGB', (self.opt.default_h, self.opt.default_w))

                label = 'error'


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

        # 0 用来padding,解码的时候记得+1
        label_code = [self.charset.index(char)+1 for char in label]
        if len(label_code)<self.max_length:
            label_code.extend([0 for _ in range(self.max_length - len(label_code))])


        label = torch.from_numpy(np.array(label_code))
        label_len = torch.from_numpy(np.array(len(label_code)))

        return im, label, label_len, width

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.num_samples

