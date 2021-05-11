from .lmdb_shuffle_dataset import LMDBShuffleDataset
from torch.utils.data import ConcatDataset
from .base_dataset import BaseDataset

def build_lmdb_datasets(cfg,paths:str,charset_str,augment=0):
    if isinstance(paths,list):
        datasets =  [LMDBShuffleDataset(cfg,path,charset_str,augment=augment) for path in paths]
        final_dataset = ConcatDataset(datasets)
    elif isinstance(paths,str):
        final_dataset = LMDBShuffleDataset(cfg,paths,charset_str,augment=augment)
    return final_dataset


class CycleCondCtcLMDBDataset(BaseDataset):
    def __init__(self,opt):
        super(CycleCondCtcLMDBDataset, self).__init__(opt)
        charset_str, _ = open_charset_file(opt.charset)
        source_trainset = build_lmdb_datasets(opt, opt.source_lmdb, charset_str, augment=0)
        target_trainset = build_lmdb_datasets(opt, opt.target_lmdb, charset_str, augment=1)
        self.ds1 = source_trainset
        self.ds2 = target_trainset

    def __len__(self):
        return max(len(self.ds1), len(self.ds2))

    def __getitem__(self, idx):
        sample = {
                "S": self.ds1[idx % len(self.ds1)][0],
                "S_labels": self.ds1[idx % len(self.ds1)][1],
                "S_label_lens": self.ds1[idx % len(self.ds1)][2],
                "S_width": self.ds1[idx % len(self.ds1)][3],
                "T": self.ds2[idx % len(self.ds2)][0],
                "T_labels": self.ds2[idx % len(self.ds2)][1],
                "T_label_lens": self.ds2[idx % len(self.ds2)][2],
                "T_width": self.ds2[idx % len(self.ds2)][3]
            }
        return sample


def open_charset_file(charset_path:str)->(str,dict):
    char_dict = {}
    with open(charset_path,'r',encoding='utf-8') as file_handler:
        lines = file_handler.readlines()
        lines = [line.rstrip() for line in lines]
        for index,line in enumerate(lines):
            char_dict[index]=line
        return "".join(lines),char_dict
