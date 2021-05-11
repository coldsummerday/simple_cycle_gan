import logging

from .lmdb_shuffle_dataset import LMDBShuffleDataset
from torch.utils.data import ConcatDataset
from .alignedlmdb_dataset import AlignedLMDBDataset
from .cyclecondctclmdb_dataset import CycleCondCtcLMDBDataset
from ..utils.logutils import print_log


registered_datasets = {
    "AlignedLMDBDataset":AlignedLMDBDataset,
    "CycleCondCtcLMDBDataset":CycleCondCtcLMDBDataset
}

def build_dataset(cfg):
    if cfg.datasettype not in registered_datasets.keys():
        print_log("the dataset type {} is not implement".format(cfg.datasettype),level=logging.ERROR)
        raise NotImplementedError
    cls = registered_datasets[cfg.datasettype]
    return cls(cfg)



