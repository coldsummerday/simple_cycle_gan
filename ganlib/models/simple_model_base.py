
import  torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        pass

    def forward(self,data:dict,return_loss:bool):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def forward_test(self,data:dict)->dict:
        raise NotImplemented

    def forward_train(self,data:dict)->dict:
        raise NotImplemented


    def optimize_parameters(self,data:dict):
        """
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        :return:
        """
        raise NotImplementedError

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
