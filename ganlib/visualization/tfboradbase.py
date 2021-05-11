import os.path

from torch.utils.tensorboard import SummaryWriter
import torch
from ..utils.ori_utils import tensor2im,save_image


def unnormalized_show(img:torch.tensor,std=0.5,mean=0.5):
    img = img * std + mean     # unnormalize
    return img


class VisualBase(object):
    def __init__(self,path:str,save_img_flag = False):
        self.tf_borad_writer = SummaryWriter(path)
        self.save_img_flag = save_img_flag
        self.save_path = path
        self.register_scale_names = []
        self.scale_group_name_dict = {}
        self.register_imgtensor_names = []

    def register_scale(self,names:[],group_name=None):
        if isinstance(names,str):
            self.register_scale_names.append(names)
            if group_name!=None:
                self.scale_group_name_dict[names]=group_name
        elif isinstance(names,list) and isinstance(names[0],str):
            self.register_scale_names.extend(names)
            if group_name!=None:
                for name in names:
                    self.scale_group_name_dict[name]=group_name
    def register_imgs_name(self,names:str):
        if isinstance(names,str):
            self.register_imgtensor_names.append(names)
        elif isinstance(names,list) and isinstance(names[0],str):
            self.register_imgtensor_names.extend(names)

    def get_scale_tag_name(self,scale_name:str):
        if scale_name in self.scale_group_name_dict.keys():
            return "{}/{}".format(self.scale_group_name_dict[scale_name],scale_name)
        else:
            return scale_name

    def write_once(self,data_dict:dict,total_iter:int):
        for scale_name in self.register_scale_names:
            if scale_name in data_dict.keys():
                scale_tag = self.get_scale_tag_name(scale_name)
                self.tf_borad_writer.add_scalar(scale_tag,data_dict[scale_name],total_iter)

        for img_tensor_name in self.register_imgtensor_names:
            if img_tensor_name in data_dict.keys():
                self.tf_borad_writer.add_images(img_tensor_name, unnormalized_show(data_dict[img_tensor_name]),global_step=total_iter,dataformats="NCHW")
                if self.save_img_flag:
                    img_name = os.path.join(self.save_path,"iter_{}_{}.jpg".format(total_iter,img_tensor_name))
                    image_numpy = tensor2im(data_dict[img_tensor_name])
                    save_image(image_numpy, img_name)








