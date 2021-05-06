# from ..base_config import *
from os.path import abspath,dirname,join
__dir__= dirname(abspath(__file__))
base_config_file = join(__dir__,"../base_config.py")

Experiment_name = "cycle_gan_cond_zhb"
Model = 'cycle_ocr_cond'
N_class =  6129

#data set
Crop_size = [32,160]
Preprocess = 'scale_width_and_crop'

UseLMDBDataset = True
Charset = './data/chinese_charset.txt' #"the char~ set for  code  each char one line"
Source_lmdb = './data/lmdbdataset/src_generate/'
Target_lmdb = './data/lmdbdataset/train_1w_zhb/'
Test_lmdb = './data/lmdbdataset/val_1w_zhb'

Pad_h_ratio = '0.05,0.15' #'padding height ratio'
Pad_w_ratio = '0.05,0.30' #'padding width ratio'
Font_obl_a = '-5,5'#'rotate angle'
Font_shade_range = '180,255' #'controlling shade of font'
Font_clarity_range = '0.5,1' #'controlling clarity of font'
Font_size_range = '25,50' #control width of font
#loss setting
Lambda_S=10.0
Lambda_T=10.0
Lambda_identity=10.0

#pretrain
Pretrain_ctc_model = './checkpoints/train_srcg/ocr-44000.pth'

# #
# Resume = None





