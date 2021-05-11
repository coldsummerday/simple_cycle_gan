# from ..base_config import *
from os.path import abspath,dirname,join
__dir__= dirname(abspath(__file__))
base_config_file = join(__dir__,"../base_config.py")

Experiment_name = "pix2pix_inpainbg_mix"
Model = 'pix2pix'
Batch_size = 512




#train setting
Save_epoch_freq = 200 #'frequency of saving checkpoints at the end of epochs'
Save_iter_freq = None
Total_epoch = 800 #total train epoch

Log_freq = 200
Visual_freq = 400


#net
#net setting
Ngf = 64 #of gen filters in the last conv layer
Ndf = 64 #of discrim filters in the first conv layer
"""
'specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator'
"""
# netd = 'n_spect_layers'
netd = 'basic'

"""
specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128 | unet_32]
"""
netg = 'unet_32'
#'only used if netD==n_layers'
N_layers_D = 1
Norm = 'batch' #'instance normalization or batch normalization [instance | batch | none]'
Init_type = 'normal' #'network initialization [normal | xavier | kaiming | orthogonal]'
Init_gain  = 0.02 #'scaling factor for normal, xavier and orthogonal.'
No_dropout = True
Skip_connect = True
Gan_mode = 'vanilla' #'the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.'

Pool_size = 0


#data set
Crop_size = [32,160]
Preprocess = 'scale_width_and_crop'
Direction = "AtoB"
DatasetType = "AlignedLMDBDataset"

Source_lmdb = './data/lmdbdataset/license_pair_bg_lmdb_data/roi_lmdb'
Target_lmdb = './data/lmdbdataset/license_pair_bg_lmdb_data/roi_bg_lmdb'
# Source_lmdb = [
# # './data/lmdbdataset/src_generate/',
# './data/lmdbdataset/pair_bg_lmdb_data/driver_front_roi_lmdb'
#
# ]
# Target_lmdb = [
#     './data/lmdbdataset/pair_bg_lmdb_data/driver_front_roi_bg_lmdb'
# ]
#loss setting
Lambda_L1 = 100.0



# #
# Resume = None





