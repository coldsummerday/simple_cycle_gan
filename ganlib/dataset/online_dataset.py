from dataset.base_dataset import BaseDataset, get_transform, get_params
from PIL import Image
import random
import torch, cv2
import numpy as np
import torchvision.transforms as transforms
from data_augment import Transform
import os
from PIL import Image, ImageFont, ImageDraw
from util.common import AddRotate, AddBlur


class OnlineDataset(BaseDataset):
    def __init__(self, opt, rev_chn_dict, augment=0, shuffle=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.input_nc = self.opt.output_nc
        self.max_length = 60
        self.shuffle = shuffle
        self.augment = augment
        self.erosion = False
        self.corpus_dir = './corpus'
        self.corpus_names = ['common_jx']
        self.font_dir = './font_ttf'
        self.font_style = ['heiti', 'songti', 'jianti', 'kaiti']
        self.rev_chn_dict = rev_chn_dict
        self.pad_h_ratio = [float(i) for i in self.opt.pad_h_ratio.split(',')]
        self.pad_w_ratio = [float(i) for i in self.opt.pad_w_ratio.split(',')]
        self.font_shade_range = [int(i) for i in self.opt.font_shade_range.split(',')]
        self.font_obl_a = [float(i) for i in self.opt.font_obl_a.split(',')]
        self.erode_level = (1, 2)
        self.chinese_sty_query, self.english_sty_query = self.build_query(self.font_style)
        self.fontChns = []
        self.fontEngs = []
        for style in self.font_style:
            self.fontChns.extend(self.chinese_sty_query[style])
            self.fontEngs.extend(self.chinese_sty_query[style])
        self.font_clarity_range = [float(i) for i in self.opt.font_clarity_range.split(',')]  # control the clarity of font
        self.font_size_range = [int(i) for i in self.opt.font_size_range.split(',')]
        print('chinese font: ', self.fontChns)
        print('english font: ', self.fontEngs)
        self.pad_labels = []
        self.real_labels = []
        self.label_lens = []
        print("\nBuilding online source datasets ... ")
        self.build_dataset()

    def build_query(self, font_style):
        chinese_sty_query = {}
        english_sty_query = {}
        for key in font_style:
            all_english_common_style, all_chinese_common_style = self.search_all_font('font_ttf/', key, ['TTF', 'OTF'])
            chinese_sty_query[key] = all_chinese_common_style
            english_sty_query[key] = all_english_common_style
        return chinese_sty_query, english_sty_query

    def search_all_font(self, filename, field, exps):
        all_english_common_style = []
        all_chinese_common_style = []
        for root, _, filenames in os.walk(filename):
            for filename in filenames:
                if not self.endswith(filename, exps):
                    continue
                if field not in root and field not in filename:
                    continue
                if 'english' in root:
                    all_english_common_style.append(os.path.join(root, filename))
                if 'chinese' in root:
                    all_chinese_common_style.append(os.path.join(root, filename))
        return all_english_common_style, all_chinese_common_style

    def endswith(self, filename, exps):
        filename = filename.upper()
        if isinstance(exps, list):
            for exp in exps:
                exp = exp.upper()
                if filename.endswith(exp):
                    return True
        else:
            if filename.endswith(exps.upper()):
                return True
        return False

    def build_dataset(self):
        for corpus in self.corpus_names:
            with open(os.path.join(self.corpus_dir, corpus, '0.txt'), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if self.shuffle:
                    random.shuffle(lines)
                for line in lines:
                    line = line.strip()
                    # label index
                    codes = []
                    for c in line:
                        if c in self.rev_chn_dict:
                            codes.append(self.rev_chn_dict[c])

                    if len(codes) > self.max_length or len(codes) == 0:
                        continue

                    img_code = [int(code) for code in codes if int(code) < 6097]
                    self.real_labels.append(line)
                    self.label_lens.append(len(img_code))
                    # 把标签索引改为等长，后面填充0
                    img_code += [0] * (self.max_length - len(img_code))
                    self.pad_labels.append(img_code)

    def GenChn(self, val, h, w, f, f_rot):
        """
        生成中文字符
        """
        fw, fh = f.getsize(val)
        img=Image.new("L", (fw, fh))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), val, (255), font=f)
        if f_rot != 0:
            img = img.rotate(f_rot, expand=True, fillcolor=(0))
        img = img.resize((w, h))
        return np.array(img)

    def GenEng(self, val, h, w, f, f_rot):
        """
        生成英文字符
        """
        size = f.getsize(val)
        img=Image.new("L", size)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), val, (255), font=f)
        if f_rot != 0:
            img = img.rotate(f_rot, expand=True, fillcolor=(0))
        img = img.resize((w, h))
        return np.array(img)

    def replace_font_color(self, bg_mean):
        font_color = []
        for i in range(3):
            color = bg_mean[i]
            if color > 128:
                color = int(2 / 3 * color)
                color = np.random.randint(0, color)
            else:
                color = int(5 / 3 * color)
                color = np.random.randint(color, 255)
            font_color.append(color)
        return font_color

    def replace_background(self, font_area, bg_img):
        x = np.random.randint(0, max(bg_img.shape[1]-font_area.shape[1], 1))
        y = np.random.randint(0, max(bg_img.shape[0]-font_area.shape[0], 1))
        if x > 0 or y > 0:
            bg_img = bg_img[y: y+font_area.shape[0], x: x+font_area.shape[1]]
        if np.any(bg_img.shape[:2] != font_area.shape[:2]):
            for i in range(2):
                scale = 1.0 * font_area.shape[i] / bg_img.shape[i]
                if scale > 2.:
                    bg_img = np.repeat(bg_img, int(scale)+1, i)
                    if i == 0:
                        bg_img = bg_img[:font_area.shape[i]]
                    else:
                        bg_img = bg_img[:,:font_area.shape[i]]
            bg_img = cv2.resize(bg_img, (font_area.shape[1], font_area.shape[0]))

        font_color = np.random.randint(0, 50)
        bg_img[font_area] = (font_color, font_color, font_color)
        return bg_img

    def generate_single_image(self, text):
        font_h = self.font_size_range[1]
        font_w = np.random.randint(self.font_size_range[0], self.font_size_range[1])
        clarity = int(font_h * np.random.uniform(self.font_clarity_range[0], self.font_clarity_range[1]))

        # randomly choose 1 type of the font style
        selected_style = random.choice(self.font_style)
        fontChns = self.chinese_sty_query[selected_style]
        fontEngs = self.english_sty_query[selected_style]
        fchn = ImageFont.truetype(random.choice(fontChns), clarity, 0)
        feng = ImageFont.truetype(random.choice(fontEngs), clarity, 0)

        # create an all black bg image
        num_char = len(text)
        char_gap = np.random.randint(3, 2 * font_w) if num_char < 5 else np.random.randint(3, 5)
        # char_gap = np.random.randint(3, 5)
        padding_w_ratio = np.random.uniform(self.pad_w_ratio[0], self.pad_w_ratio[1], 2)
        padding_w = [int(font_h * padding_w_ratio[0]), int(font_h * padding_w_ratio[1])]
        total_w = padding_w[0] + font_w * num_char + char_gap * (num_char - 1) + padding_w[1]

        padding_h_ratio = np.random.uniform(self.pad_h_ratio[0], self.pad_h_ratio[1], 2)
        padding_h = [int(font_h * padding_h_ratio[0]), int(font_h * padding_h_ratio[1])]
        total_h = padding_h[0] + font_h + padding_h[1]

        new_img = np.zeros((total_h, total_w))

        # cover the bg image with characters
        start_h = int(np.random.random() * (total_h - font_h))
        start_w = int(np.random.random() * sum(padding_w))
        if self.erosion:
            erode_level = np.random.randint(self.erode_level[0], self.erode_level[1])
        else:
            erode_level = None

        if np.random.random() < 0.5:
            f_rot = np.random.randint(self.font_obl_a[0], self.font_obl_a[1])
        else:
            f_rot = 0

        for it in text:
            if u'\u4e00' <= it <= u'\u9fa5':
                tmp = self.GenChn(it, font_h, font_w, fchn, f_rot)
            else:
                tmp = self.GenEng(it, font_h, font_w, feng, f_rot)

            if erode_level is not None:
                kernel = np.ones((erode_level, erode_level), np.uint8)
                tmp = cv2.erode(tmp, kernel)

            new_img[start_h:start_h+font_h, start_w:start_w+font_w] = tmp
            start_w += font_w + char_gap

        if np.random.random() < 0.5:
            if len(text) > 25:
                rotate_rot = np.random.uniform(0.5, 1.0)
            else:
                rotate_rot = np.random.uniform(0.5, 1.5)
            new_img = AddRotate(new_img, bg_color=(0,0), rot=rotate_rot)

        new_img = np.tile(np.expand_dims(new_img, axis=2), (1, 1, 3))
        font_shade = 512 - np.random.randint(self.font_shade_range[0], self.font_shade_range[1])
        new_img = (font_shade - new_img).clip(0, 128)
        new_img = new_img.astype(np.uint8)
        return new_img

    def __getitem__(self, index):
        """Return a dataset point and its metadata information.
        Parameters:
            index (int)      -- a random integer for dataset indexing
        """
        text = self.real_labels[index]
        img = self.generate_single_image(text)
        img = Image.fromarray(img)

        # 数据增强
        if self.augment:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            trans = Transform(img)
            img = trans.data_augment()
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # apply image transformation
        params = get_params(self.opt, img.size)
        transform_list = get_transform(self.opt, params=params, grayscale=(self.input_nc == 1))
        img = transform_list[0](img)
        width = img.size[0]
        if width > 640:
            width = 640
        transform = transforms.Compose(transform_list[1:])
        im = transform(img)


        label = torch.from_numpy(np.array(self.pad_labels[index]))
        label_len = torch.from_numpy(np.array(self.label_lens[index]))

        return im, label, label_len, width

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.real_labels)
