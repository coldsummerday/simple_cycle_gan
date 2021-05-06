"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import logging
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
import sys
import os
import argparse
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(__dir__,"../"))



from ganlib.utils import concat_dataset,dict_data2model
from ganlib.dataset.lmdb_shuffle_dataset import LMDBShuffleDataset
from ganlib.models.cycle_ocr_cond_model import CycleOCRCondModel
from ganlib.models import create_model
from ganlib.utils import Config,print_log,get_print_cfg_options,get_root_logger,load_checkpoint,save_checkpoint
from ganlib.utils.dist_utils import init_dist

def parse_args():
    parser = argparse.ArgumentParser(description='simple_cycle_gan train ')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--resume', help='the checkpoint file to resume from')

    parser.add_argument("--distributed", default=1, type=int, help="use DistributedDataParallel to train")
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    options_message = get_print_cfg_options(cfg)
    charset_str,charset_dict = open_charset_file(cfg.charset)
    os.makedirs(cfg.checkpoints_dir,exist_ok=True)
    experiment_dir = os.path.join(cfg.checkpoints_dir,cfg.experiment_name)
    os.makedirs(experiment_dir,exist_ok=True)
    cfg.is_train = True

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(experiment_dir, '{}.log'.format(timestamp))
    ##pytorch 1.8 Pytorch 1.8 distributed mode will disable python logging module
    """
    解决方法：
     Because during the execution of dist.init_process_group,
      it will call _store_based_barrier, which finnaly will call logging.info (see the source code here). So if you call logging.basicConfig before you call dist.init_process_group
    即logging.basicConfig(format=format_str, level=log_level) 调用在 torch.init_process_group之前
    """
    logger = get_root_logger(log_file=log_file, log_level=logging.INFO)
    logger.info(options_message)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = create_model(cfg)
    if args.distributed and torch.cuda.is_available() and torch.cuda.device_count()>1:
        init_dist("pytorch",backend='nccl')
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    else:
        model = model.to(device)


    testset = LMDBShuffleDataset(cfg, cfg.test_lmdb, charset_str, charset_dict)
    testloader = DataLoader(testset, cfg.batch_size, shuffle=False, num_workers=cfg.num_threads)

    Source_trainset = LMDBShuffleDataset(cfg, cfg.source_lmdb, charset_str, augment=0)
    Target_trainset = LMDBShuffleDataset(cfg, cfg.target_lmdb, charset_str, augment=1)
    ConcateDataset = concat_dataset(Source_trainset, Target_trainset)
    dataset_size = len(ConcateDataset)  # get the number of images in the dataset.
    logger.info('The number of training images = %d' % dataset_size)
    trainloader = torch.utils.data.DataLoader(ConcateDataset,
                                              cfg.batch_size,
                                              shuffle=True,
                                              num_workers=cfg.num_threads)

    start_num_iter = 0
    start_num_epoch = 0
    if args.resume is not None:
        checkpoint=load_checkpoint(model,cfg.resume,device)
        if "num_epoch" in checkpoint.keys():
            start_num_epoch = checkpoint["num_epoch"]
            start_num_iter = checkpoint["num_iter"]
        logger.info("model load checkpoint:{},start_epoch:{},start_iter:{}".format(cfg.resume,start_num_epoch,start_num_iter))

    total_iters = 0
    for epoch in range(start_num_epoch,cfg.total_epoch):

        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(trainloader):  # inner loop within one epoch
            total_iters += 1
            epoch_iter += 1

            #格式化数据
            data_dict = model.set_input_dict(data)
            data_dict = dict_data2model(model, data_dict)
            netC_T, netC_S, netG_T, netG_S, netD_T, netD_S, loss_dict = model.forward(data_dict, return_loss=True)
            print(loss_dict)
            if epoch_iter % cfg.log_freq ==0:
                #TODO:logging
                pass

        if (epoch+1) % cfg.save_epoch_freq==0:
            save_checkpoint(model=model,filename=os.path.join(experiment_dir,"{}_full.pth".format(epoch)),num_epoch=epoch,num_iter=total_iters)
            torch.save(netG_T.state_dict(), os.path.join(experiment_dir,"{}_GT.pth".format(epoch)))
            torch.save(netG_S.state_dict(), os.path.join(experiment_dir,"{}_GS.pth".format(epoch)))

def open_charset_file(charset_path:str)->(str,dict):
    char_dict = {}
    with open(charset_path,'r') as file_handler:
        lines = file_handler.readlines()
        lines = [line.rstrip() for line in lines]
        for index,line in enumerate(lines):
            char_dict[index]=line
        return "".join(lines),char_dict

if __name__ == '__main__':
    main()