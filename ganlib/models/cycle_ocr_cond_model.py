import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import itertools
from .simple_model_base import BaseModel
from . import networks
from ..utils import concat_labels
from ..utils.image_pool import ImagePool
from ..utils.dist_utils import master_only
from ..utils import save_checkpoint

from ..visualization.tfboradbase import VisualBase
class CycleOCRCondModel(BaseModel):
    """
       This class implements the CycleGAN model, for learning image-to-image translation without paired dataset.

       The model training requires '--dataset_mode unaligned' dataset.
       By default, it uses a '--netG resnet_9blocks' ResNet generator,
       a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
       and a least-square GANs objective ('--gan_mode lsgan').

       CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self,opt):
        super(CycleOCRCondModel, self).__init__()
        self.loss_names = ['D_S', 'D_T', 'G_S', 'G_T', 'cycle_S', 'cycle_T', 'ctc_S', 'ctc_T', 'ctc_fS', 'ctc_fT',
                           'ctc_cS', 'ctc_cT']

        self.visual_loss_name_dict = {
            "G":[ "loss_G","loss_G_S","loss_G_T","loss_cycle_T","loss_cycle_S","loss_idt_S","loss_idt_T"],
            "D":["loss_D","loss_D_T","loss_D_S"],
            "CTC":[ "loss_ctc","loss_ctc_S","loss_ctc_T","loss_ctc_fS","loss_ctc_fT"]
        }
        self.visual_tensor_name = ["input_S","input_S","fake_T","fake_S"]



        self.opt = opt
        self.is_train = opt.is_train
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # concate A and B
        self.netG_T = networks.define_G(6, opt.output_nc, opt.ngf, opt.netg, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain,
                                        skip_connect=opt.skip_connect)
        self.netG_S = networks.define_G(6, opt.output_nc, opt.ngf, opt.netg, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain,
                                        skip_connect= opt.skip_connect)
        self.netC_S = networks.define_OCR(opt.n_class,opt.crop_size ,opt.init_type, opt.init_gain)
        self.netC_T = networks.define_OCR(opt.n_class,opt.crop_size,opt.init_type, opt.init_gain)

        if self.opt.pretrain_ctc_model is not None:
            self.netC_S.load_state_dict(
                torch.load(self.opt.pretrain_ctc_model, map_location=device))
            self.netC_T.load_state_dict(
                torch.load(self.opt.pretrain_ctc_model, map_location=device))

        if self.is_train:
            self.netD_S = networks.define_D(opt.output_nc, opt.ndf, opt.netd,
                                            opt.n_layers_d, opt.norm, opt.init_type, opt.init_gain)
            self.netD_T = networks.define_D(opt.input_nc, opt.ndf, opt.netd,
                                            opt.n_layers_d, opt.norm, opt.init_type, opt.init_gain)

        #define_loss_func
        if self.is_train:
            # define loss functions
            self.loss_func_GAN = networks.GANLoss(opt.gan_mode).to(device)  # define GAN loss.
            self.loss_func_Cycle = torch.nn.L1Loss()
            self.loss_func_Idt = torch.nn.L1Loss()
            self.loss_func_CTC = torch.nn.CTCLoss(blank=0, reduction='none')

            self.fake_S_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_T_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            h_down_scale = self.netC_S.h_down_scale
            w_down_scale = self.netC_S.w_down_scale
            max_len = int(opt.crop_size[0] / h_down_scale) * int(opt.crop_size[1] / w_down_scale) - 1
            self.ctc_train_len = max_len
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_S.parameters(), self.netG_T.parameters(), self.netC_S.parameters(),
                                self.netC_T.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_S.parameters(), self.netD_T.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))



    def set_input_dict(self,data_dict:dict)->dict:
        input_S = data_dict['S']
        input_T = data_dict['T']
        data_dict["input_S"] = input_S
        data_dict["input_T"] = input_T
        data_dict["gt_S"] = data_dict["S"]
        data_dict["gt_T"] = data_dict["T"]
        return data_dict

    def forward_test(self,data:dict) ->dict:
        assert  set(["input_S","input_T"]).issubset(data.keys())
        input_S = data["input_S"]
        input_T = data["input_T"]
        fake_T = self.netG_T(torch.cat((input_S, input_T), 1))
        fake_S = self.netG_S(torch.cat((input_T, input_S), 1))

        rec_T = self.netG_T(torch.cat((fake_S, input_T), 1))
        rec_S = self.netG_S(torch.cat((fake_T, input_S), 1))

        data["fake_T"]=fake_T
        data["fake_S"] = fake_S
        data["rec_T"] = rec_T
        data["rec_S"] = rec_S
        return data

    def forward_train(self,data:dict) ->dict:
        assert set(["input_S", "input_T"]).issubset(data.keys())
        data_dict =self.forward_test(data)
        return self.optimize_parameters(data_dict)

    def calculate_losses_G(self,data_dict:dict):
        assert set(["input_S","input_T","gt_S","gt_T","fake_S","fake_T","rec_S","rec_T"]).issubset(data_dict.keys())

        input_S = data_dict["input_S"]
        input_T = data_dict["input_T"]
        gt_S = data_dict["gt_S"]
        gt_T = data_dict["gt_T"]
        fake_S = data_dict["fake_S"]
        fake_T = data_dict["fake_T"]
        rec_S = data_dict["rec_S"]
        rec_T = data_dict["rec_T"]


        lambda_idt = self.opt.lambda_identity
        lambda_S = self.opt.lambda_s
        lambda_T = self.opt.lambda_t
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_T = self.netG_T(torch.cat((input_T, input_T), 1))
            loss_idt_T = self.loss_func_Idt(idt_T, gt_T) * lambda_T * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_S = self.netG_S(torch.cat((input_S, input_S), 1))
            loss_idt_S = self.loss_func_Idt(idt_S, gt_S) * lambda_S * lambda_idt
        else:
            loss_idt_T = 0
            loss_idt_S = 0

        loss_G_S = self.loss_func_GAN(self.netD_S(fake_S),True)

        loss_G_T = self.loss_func_GAN(self.netD_T(fake_T),True)
        loss_cycle_S = self.loss_func_Cycle(rec_S,gt_S) * lambda_S
        loss_cycle_T = self.loss_func_Cycle(rec_T,gt_T) * lambda_T
        # combined loss and calculate gradients
        loss_G = (loss_G_S + loss_G_T + loss_cycle_S + loss_cycle_T +
                       loss_idt_S + loss_idt_T) / 1.
        loss_dict = dict(
            loss_G=loss_G,
            loss_G_S = loss_G_S,
            loss_G_T = loss_G_T,
            loss_cycle_T = loss_cycle_T,
            loss_cycle_S = loss_cycle_S,
            loss_idt_S = loss_idt_S,
            loss_idt_T = loss_idt_T
        )
        return loss_dict


    def calculate_losses_CTC(self,data_dict:dict):
        S_labels = data_dict['S_labels']
        T_labels = data_dict['T_labels']
        S_label_lens = data_dict['S_label_lens']
        T_label_lens = data_dict['T_label_lens']
        gt_S = data_dict["gt_S"]
        gt_T = data_dict["gt_T"]
        input_T = data_dict["input_T"]

        fake_S = data_dict["fake_S"]
        fake_T = data_dict["fake_T"]
        rec_S = data_dict["rec_S"]
        rec_T = data_dict["rec_T"]

        device = S_labels.device

        S_labels = concat_labels(S_labels, S_label_lens).to(device)
        T_labels = concat_labels(T_labels, T_label_lens).to(device)


        S_mask = torch.ones(gt_S.size()).to(device)
        T_mask = torch.ones(input_T.size()).to(device)
        # real width
        S_width = data_dict['S_width']
        T_width = data_dict['T_width']
        for i in range(gt_S.size(0)):
            S_mask[i][:, :, S_width[i]:] = 0
        for i in range(gt_T.size(0)):
            T_mask[i][:, :, T_width[i]:] = 0

        S_input_len = torch.full((gt_S.shape[0],), self.ctc_train_len, dtype=torch.long)

        T_input_len = torch.full((gt_T.shape[0],), self.ctc_train_len, dtype=torch.long)

        ##TODO:这里应该做inf替换,不然backward全是nan
        loss_ctc_S = self.loss_func_CTC(self.netC_S(gt_S * S_mask).log_softmax(-1), S_labels,
                                            S_input_len, S_label_lens).mean()
        loss_ctc_T = self.loss_func_CTC(self.netC_T(gt_T * T_mask).log_softmax(-1), T_labels,
                                            T_input_len, T_label_lens).mean()
        loss_ctc_fS = self.loss_func_CTC(self.netC_S(fake_S * T_mask).log_softmax(-1), T_labels,
                                             T_input_len, T_label_lens).mean()
        loss_ctc_fT = self.loss_func_CTC(self.netC_T(fake_T * S_mask).log_softmax(-1), S_labels,
                                             S_input_len, S_label_lens).mean()
        loss_ctc_cS = self.loss_func_CTC(self.netC_S(rec_S * S_mask).log_softmax(-1), S_labels,
                                             S_input_len, S_label_lens).mean()
        loss_ctc_cT = self.loss_func_CTC(self.netC_T(rec_T * T_mask).log_softmax(-1), T_labels,
                                             T_input_len, T_label_lens).mean()
        loss_ctc = (loss_ctc_S + loss_ctc_T + loss_ctc_fS + loss_ctc_fT +
                         loss_ctc_cS + loss_ctc_cT) / 1.
        return dict(
            loss_ctc=loss_ctc,
            loss_ctc_S=loss_ctc_S,
            loss_ctc_T=loss_ctc_T,
            loss_ctc_fS=loss_ctc_fS,
            loss_ctc_fT = loss_ctc_fT,
            loss_ctc_cS=loss_ctc_cS,
            loss_ctc_cT = loss_ctc_cT
        )

    def calculate_losses_D(self,data_dict:dict):

        fake_S = data_dict["fake_S"]
        fake_T = data_dict["fake_T"]

        gt_S = data_dict["gt_S"]
        gt_T = data_dict["gt_T"]

        fake_S = self.fake_S_pool.query(fake_S)
        loss_D_S = self.backward_D_basic(self.netD_S,gt_S,fake_S)

        fake_T = self.fake_T_pool.query(fake_T)
        loss_D_T = self.backward_D_basic(self.netD_T, gt_T, fake_T)
        loss_D = (loss_D_S + loss_D_T) / 1.
        return dict(
            loss_D = loss_D,
            loss_D_T = loss_D_T,
            loss_D_S = loss_D_S
        )

    def backward_D_basic(self, netD:nn.Module, real:torch.Tensor, fake:torch.Tensor):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.loss_func_GAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.loss_func_GAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def optimize_parameters(self,data:dict):
        """
        after forward_test
        :param data:
        :return:
        """
        total_loss_dict = {}
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # G_A and G_B
        self.set_requires_grad([self.netD_S, self.netD_T], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        # calculate gradients for G_A and G_B
        loss_dict_G = self.calculate_losses_G(data_dict=data)
        total_loss_dict.update(loss_dict_G)

        loss_dict_ctc = self.calculate_losses_CTC(data_dict=data)
        # total_loss_dict.update(loss_dict_ctc)
        if torch.isinf(loss_dict_ctc["loss_ctc"]) or torch.isnan(loss_dict_ctc["loss_ctc"]):
            loss_dict_G["loss_G"].backward()
        else:
            (loss_dict_G["loss_G"]+loss_dict_ctc["loss_ctc"]).backward()
        clip_grad_norm_(itertools.chain(self.netG_T.parameters(), self.netG_S.parameters(), self.netC_S.parameters(),
                                        self.netC_T.parameters()), 20)
        self.optimizer_G.step()  # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_S, self.netD_T], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero

        loss_dict_D = self.calculate_losses_D(data_dict=data)
        total_loss_dict.update(loss_dict_D)
        loss_dict_D["loss_D"].backward()
        clip_grad_norm_(itertools.chain(self.netD_S.parameters(), self.netD_T.parameters()), 20)
        self.optimizer_D.step()  # update D_A and D_B's weights
        return data,total_loss_dict
        # return self.netC_T, self.netC_S, self.netG_T, self.netG_S, self.netD_T, self.netD_S,total_loss_dict

    def register_scale_img_2_visual(self,visual_handler:VisualBase):
        for key,value in self.visual_loss_name_dict.items():
            visual_handler.register_scale(value,key)
        visual_handler.register_imgs_name(self.visual_tensor_name)

    @master_only
    def save_checkpoints(self,save_paths:str,num_epoch:int,num_iter:int):
        save_checkpoint(model=self, filename=os.path.join(save_paths, "{}_{}_full.pth".format(num_epoch,num_iter)), num_epoch=num_epoch,
                    num_iter=num_iter)
        torch.save(self.netG_T.state_dict(), os.path.join(save_paths, "{}_{}_GT.pth".format(num_epoch,num_iter)))
        torch.save(self.netG_S.state_dict(), os.path.join(save_paths, "{}_{}_GS.pth".format(num_epoch,num_iter)))














        
        