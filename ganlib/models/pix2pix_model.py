import os
import torch
from .simple_model_base import BaseModel
from . import networks
from ..utils.dist_utils import master_only
from ..utils import save_checkpoint
from ..visualization.tfboradbase import VisualBase


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

        The model training requires '--dataset_mode aligned' dataset.
        By default, it uses a '--netG unet256' U-Net generator,
        a '--netD basic' discriminator (PatchGAN),
        and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

        pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
        """

    def __init__(self,opt):
        super(Pix2PixModel, self).__init__()
        self.opt = opt

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc,opt.output_nc,opt.ngf,opt.netg,
                                      opt.norm,not opt.no_dropout,opt.init_type,opt.init_gain)
        self.is_train = opt.is_train
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.is_train:
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc+opt.output_nc,opt.ndf,opt.netd,
                                          opt.n_layers_d,opt.norm,opt.init_type,opt.init_gain)

            # define loss functions
            self.loss_func_GAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.loss_func_L1 = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.visual_loss_name_dict = {
                "G": ["loss_G","loss_G_GAN", "loss_G_L1"],
                "D": ["loss_D","loss_D_real", "loss_D_fake"],
            }
            self.visual_tensor_name = ["fake_B","real_A","real_B"]
    def set_input_dict(self,data_dict:dict)->dict:
        assert  set(["A","B"]).issubset(data_dict.keys())

        AtoB = self.opt.direction == 'AtoB'
        data_dict["real_A"] = data_dict["A" if AtoB else "B"]
        data_dict["real_B"] = data_dict["B" if AtoB else "A"]
        return data_dict

    def forward_test(self,data:dict) ->dict:
        assert  set(["real_A"]).issubset(data.keys())

        fake_B = self.netG(data["real_A"])
        data["fake_B"] = fake_B
        return data

    def forward_train(self,data:dict) ->dict:
        assert  set(["real_A","real_B"]).issubset(data.keys())
        data_dict = self.forward_test(data)
        return self.optimize_parameters(data_dict)


    def calculate_losses_G(self,data_dict:dict):
        """Calculate GAN and L1 loss for the generator"""
        real_A = data_dict["real_A"]
        fake_B =  data_dict["fake_B"]
        real_B = data_dict["real_B"]
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_A,fake_B),dim=1)
        pred_fake = self.netD(fake_AB)
        loss_G_GAN = self.loss_func_GAN(pred_fake,True)
        # Second, G(A) = B
        loss_G_L1 = self.loss_func_L1(fake_B,real_B) * self.opt.lambda_l1

        loss_G = loss_G_GAN + loss_G_L1
        return dict(
            loss_G = loss_G,
            loss_G_L1 = loss_G_L1,
            loss_G_GAN = loss_G_GAN,
        )

    def calculate_losses_D(self,data_dict:dict):
        """Calculate GAN loss for the discriminator"""
        real_A = data_dict["real_A"]
        fake_B = data_dict["fake_B"]
        real_B = data_dict["real_B"]

        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((real_A,fake_B),dim=1)
        pred_fake = self.netD(fake_AB.detach())
        loss_D_fake = self.loss_func_GAN(pred_fake,False)


        #real
        real_AB = torch.cat((real_A,real_B),dim=1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.loss_func_GAN(pred_real,True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return dict(
            loss_D = loss_D,
            loss_D_real = loss_D_real,
            loss_D_fake = loss_D_fake
        )
    def optimize_parameters(self, data_dict:dict):
        """
        after forward_test
        :param data_dict:
        :return:
        """
        total_loss_dict = {}
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # update D
        self.set_requires_grad(self.netD,True)
        self.optimizer_D.zero_grad()
        loss_dict_D = self.calculate_losses_D(data_dict=data_dict)
        total_loss_dict.update(loss_dict_D)
        loss_dict_D["loss_D"].backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD,False)
        self.optimizer_G.zero_grad()
        loss_dict_G = self.calculate_losses_G(data_dict=data_dict)
        total_loss_dict.update(loss_dict_G)
        loss_dict_G["loss_G"].backward()
        self.optimizer_G.step()
        return data_dict,total_loss_dict

    def register_scale_img_2_visual(self,visual_handler:VisualBase):
        for key,value in self.visual_loss_name_dict.items():
            visual_handler.register_scale(value,key)
        visual_handler.register_imgs_name(self.visual_tensor_name)

    @master_only
    def save_checkpoints(self, save_paths: str, num_epoch: int, num_iter: int):
        save_checkpoint(model=self, filename=os.path.join(save_paths, "{}_{}_full.pth".format(num_epoch,num_iter)), num_epoch=num_epoch,
                    num_iter=num_iter)
        torch.save(self.netG.state_dict(), os.path.join(save_paths, "{}_{}_G.pth".format(num_epoch,num_iter)))







