"""
基本配置选项,大写开头的会写入到namespcae的属性名字
小写开头的为暂时变量

"""

Experiment_name = "experiment_name" ##'name of the experiment. It decides where to store samples and models'
Checkpoints_dir = "./checkpoints" #models are saved here
Model = "base" #help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
Batch_size = 8
Input_nc = 3 #of input image channels: 3 for RGB and 1 for grayscale
Output_nc = 3 #of output image channels: 3 for RGB and 1 for grayscale
Crop_size = [32,640]

#net setting



Ngf = 64 #of gen filters in the last conv layer
Ndf = 64 #of discrim filters in the first conv layer

"""
'specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator'
"""
netd = 'n_spect_layers'

"""
specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128 | unet_32]
"""
netg = 'unet_32'

#'only used if netD==n_layers'
N_layers_D = 1
Norm = 'instance' #'instance normalization or batch normalization [instance | batch | none]'
Init_type = 'normal' #'network initialization [normal | xavier | kaiming | orthogonal]'
Init_gain  = 0.02 #'scaling factor for normal, xavier and orthogonal.'
No_dropout = True
Skip_connect = True
Gan_mode = 'lsgan' #'the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.'

##dataset
Direction = "AtoB" # 'AtoB or BtoA'  means sources to target
Num_threads = 4 #' threads for loading dataset'
Preprocess = 'scale_width_and' # 'scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
No_flip = 1 #'if specified, do not flip the images for dataset augmentation'
Load_size =32

#train setting
Save_epoch_freq = 5 #'frequency of saving checkpoints at the end of epochs'
Total_epoch = 20 #total train epoch

Log_freq = 1000


Niter = 20 #'# of iter at starting learning rate'
Niter_decay = 10 #'# of iter to linearly decay learning rate to zero'
Beta1 = 0.5 #'momentum term of adam'
Lr =  0.0002 # 'initial learning rate for adam'
Pool_size = 50 #'the size of image buffer that stores previously generated images'
Lr_policy = 'linear' #'learning rate policy. [linear | step | plateau | cosine]'




