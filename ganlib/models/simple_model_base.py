import  torch.nn as nn
from ..visualization.tfboradbase import VisualBase
from abc import ABC,abstractmethod


class BaseModel(nn.Module,ABC):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def forward(self,data:dict,return_loss:bool):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    @abstractmethod
    def forward_test(self,data:dict)->dict:
        pass

    @abstractmethod
    def forward_train(self,data:dict)->dict:
        pass

    @abstractmethod
    def optimize_parameters(self,data:dict):
        """
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        :return:data_dict,loss_dict
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoints(self, save_paths: str, num_epoch: int, num_iter: int):
        raise NotImplemented

    @abstractmethod
    def register_scale_img_2_visual(self, visual_handler: VisualBase):
        raise NotImplemented


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
