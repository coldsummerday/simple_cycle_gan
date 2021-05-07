from torch.utils.tensorboard import SummaryWriter


class VisualBase(object):
    def __init__(self,path:str):
        self.tf_borad_writer = SummaryWriter(path)

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
                self.tf_borad_writer.add_images(img_tensor_name,data_dict[img_tensor_name],global_step=total_iter,dataformats="NCHW")







