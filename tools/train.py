import logging
import time
import torch
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torch.nn.parallel import DistributedDataParallel
import sys
import os
import argparse
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(__dir__,"../"))



from ganlib.utils import concat_dataset,dict_data2model
from ganlib.dataset import build_dataset
from ganlib.dataset.lmdb_shuffle_dataset import LMDBShuffleDataset
from ganlib.models import create_model
from ganlib.utils import Config,print_log,get_print_cfg_options,get_root_logger,load_checkpoint,save_checkpoint
from ganlib.utils.dist_utils import init_dist
from ganlib.visualization.tfboradbase import VisualBase

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
    os.makedirs(cfg.checkpoints_dir,exist_ok=True)
    experiment_dir = os.path.join(cfg.checkpoints_dir,cfg.experiment_name)
    os.makedirs(experiment_dir,exist_ok=True)
    visual_dir = os.path.join(experiment_dir,"visual_log")
    os.makedirs(visual_dir,exist_ok=True)
    cfg.is_train = True

    visual_tfboard_handler = VisualBase(visual_dir)

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

    model.register_scale_img_2_visual(visual_tfboard_handler)


    if args.distributed and torch.cuda.is_available() and torch.cuda.device_count()>1:
        init_dist("pytorch",backend='nccl')
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    else:
        model = model.to(device)

    # testset = build_lmdb_datasets(cfg,cfg.test_lmdb,charset_str,augment=0)
    # testloader = DataLoader(testset, cfg.batch_size, shuffle=False, num_workers=cfg.num_threads)
    # Source_trainset = build_lmdb_datasets(cfg, cfg.source_lmdb, charset_str,augment=0)
    # Target_trainset = build_lmdb_datasets(cfg, cfg.target_lmdb, charset_str,augment=1)
    # # Source_trainset = LMDBShuffleDataset(cfg, cfg.source_lmdb, charset_str, augment=0)
    # # Target_trainset = LMDBShuffleDataset(cfg, cfg.target_lmdb, charset_str, augment=1)
    # ConcateDataset = concat_dataset(Source_trainset, Target_trainset)
    # dataset_size = len(ConcateDataset)  # get the number of images in the dataset.
    train_dataset = build_dataset(cfg)
    dataset_size  = len(train_dataset)
    #TODO:build dataset
    logger.info('The number of training images = %d' % dataset_size)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              cfg.batch_size,
                                              shuffle=True,
                                              num_workers=cfg.num_threads)
    logger.info('batch iters = {}'.format(len(trainloader)))
    start_num_iter = 0
    start_num_epoch = 0
    if args.resume is not None:
        checkpoint=load_checkpoint(model,args.resume,device)
        if "num_epoch" in checkpoint.keys():
            start_num_epoch = checkpoint["num_epoch"]
            start_num_iter = checkpoint["num_iter"]
        logger.info("model load checkpoint:{},start_epoch:{},start_iter:{}".format(args.resume,start_num_epoch,start_num_iter))



    total_iters = start_num_iter
    epoch_iters = len(trainloader)
    for epoch in range(start_num_epoch,cfg.total_epoch):

        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(trainloader):  # inner loop within one epoch
            total_iters += 1
            epoch_iter += 1

            #格式化数据
            data_dict = model.set_input_dict(data)
            data_dict = dict_data2model(model, data_dict)
            data_dict, loss_dict = model.forward(data_dict, return_loss=True)

            if epoch_iter % cfg.log_freq ==0:
                loss_line=print_loss_line(loss_dict)
                logger.info("epoch:{} {}/{} {}".format(epoch,epoch_iter,epoch_iters,loss_line))
            if total_iters % cfg.visual_freq == 0:
                visual_tfboard_handler.write_once(loss_dict,total_iters)
                visual_tfboard_handler.write_once(data_dict,total_iters)

            if cfg.save_iter_freq!=None and total_iters % cfg.save_iter_freq ==0:
                if hasattr(model, "module"):
                    model.module.save_checkpoints(experiment_dir, epoch, total_iters)
                else:
                    model.save_checkpoints(experiment_dir, epoch, total_iters)



        if (epoch+1) % cfg.save_epoch_freq==0:

            if hasattr(model,"module"):
                model.module.save_checkpoints(experiment_dir,epoch,total_iters)
            else:
                model.save_checkpoints(experiment_dir,epoch,total_iters)
            # save_checkpoint(model=model,filename=os.path.join(experiment_dir,"{}_full.pth".format(epoch)),num_epoch=epoch,num_iter=total_iters)
            # torch.save(netG_T.state_dict(), os.path.join(experiment_dir,"{}_GT.pth".format(epoch)))
            # torch.save(netG_S.state_dict(), os.path.join(experiment_dir,"{}_GS.pth".format(epoch)))



# def build_lmdb_datasets(cfg,paths:str,charset_str,augment=0):
#     if isinstance(paths,list):
#         datasets =  [LMDBShuffleDataset(cfg,path,charset_str,augment=augment) for path in paths]
#         final_dataset = ConcatDataset(datasets)
#     elif isinstance(paths,str):
#         final_dataset = LMDBShuffleDataset(cfg,paths,charset_str,augment=augment)
#     return final_dataset
#
# def open_charset_file(charset_path:str)->(str,dict):
#     char_dict = {}
#     with open(charset_path,'r',encoding='utf-8') as file_handler:
#         lines = file_handler.readlines()
#         lines = [line.rstrip() for line in lines]
#         for index,line in enumerate(lines):
#             char_dict[index]=line
#         return "".join(lines),char_dict

def print_loss_line(loss_dict:dict)->str:
    ##收集各项loss,方便输出
    log_str_list = []
    for loss_name, loss_value in loss_dict.items():
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.mean().item()

            log_str_list.append("{}:{:.4f}".format(loss_name,loss_value))
        # elif isinstance(loss_value, list):
        #     log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor '.format(loss_name))
    return ",".join(log_str_list)


if __name__ == '__main__':
    main()