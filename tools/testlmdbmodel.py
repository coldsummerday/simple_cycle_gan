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
import time
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(__dir__,"../"))
from ganlib.utils import concat_dataset,dict_data2model
from ganlib.dataset.lmdb_shuffle_dataset import LMDBShuffleDataset
from ganlib.options.train_options import TrainOptions
from ganlib.models.cycle_ocr_cond_model import CycleOCRCondModel




def open_charset_file(charset_path:str)->(str,dict):
    char_dict = {}
    with open(charset_path,'r') as file_handler:
        lines = file_handler.readlines()
        lines = [line.rstrip() for line in lines]
        for index,line in enumerate(lines):
            char_dict[index]=line
        return "".join(lines),char_dict

def opts_path_join(opt:TrainOptions):
    for key in dir(opt):
        if "__" ==key[:2]:
            #自带方法
            continue
        value = getattr(opt,key)
        if isinstance(value,str) and value[:2]=="./":
            new_value = os.path.join(__dir__,"../",value)

            setattr(opt,key,new_value)
    return opt



if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.dataroot=os.path.join(os.getcwd(), opt.dataroot).replace('./','')
    opt = opts_path_join(opt)
    charset_str,charset_dict = open_charset_file(opt.charset)
    opt.is_train = True

    ##decode
    #decoder_label = [charset_dict[code - 1] for code in label_code if code != 0]



    # testset = ShuffleDataset(opt, opt.test_list)
    testset = LMDBShuffleDataset(opt,opt.test_lmdb,charset_str,charset_dict)
    testloader = DataLoader(testset, opt.batch_size, shuffle=False, num_workers=0)

    model = CycleOCRCondModel(opt)
    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iteration
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

    best_line_acc = 0.0
    best_char_acc = 0.0
    best_iters = 0
    t_data = 0.0
    if opt.restore:
        with open(os.path.join(opt.checkpoints_dir, opt.name, 'state_file.txt'), 'r') as f:
            total_iters = int(f.readline().strip().split()[-1])
    else:
        with open(os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt'), 'w') as f:
            f.write("!!!Start to Train: \n\n")

    print("\n************************************")
    print("All the training state and test state can be found on %s" % os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt'))
    print("************************************\n")

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        # Source_trainset = ShuffleDataset(opt, opt.source_list, augment=0, shuffle=True)
        # Target_trainset = ShuffleDataset(opt, opt.target_list, augment=1, shuffle=True)
        Source_trainset=LMDBShuffleDataset(opt, opt.source_lmdb,charset_str,augment=0)
        Target_trainset = LMDBShuffleDataset(opt, opt.target_lmdb, charset_str,augment=1)
        ConcateDataset = concat_dataset(Source_trainset, Target_trainset)
        dataset_size = len(ConcateDataset)  # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)
        trainloader = DataLoader(ConcateDataset, opt.batch_size, shuffle=True, num_workers=0)
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for dataset loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(trainloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            data_dict = model.set_input_dict(data)
            data_dict=dict_data2model(model,data_dict)
            netC_T, netC_S, netG_T, netG_S, netD_T, netD_S,loss_dict=model.forward(data_dict,return_loss=True)
            print(loss_dict)
            # netC_T, netC_S, netG_T, netG_S, netD_T, netD_S = model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            #     save_result = total_iters % opt.update_html_freq == 0
            #     model.compute_visuals()
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            #
            # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            #     losses = model.get_current_losses()
            #     t_comp = (time.time() - iter_start_time) / opt.batch_size
            #     visualizer.print_current_losses(epoch_iter, losses, t_comp, t_data)
            #     if opt.display_id > 0:
            #         visualizer.plot_current_losses(total_iters, losses)

            # # test and save the best CTC model
        #     if total_iters % opt.test_freq == 0:
        #         print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        #         save_suffix = 'latest'
        #         model.save_networks(save_suffix)
        #     #     print('***********start to test*************')
        #     #     netC_T.eval()
        #     #     line_acc, char_acc = test(netC_T, opt, device, charset_dict)
        #     #     netC_T.train()
        #     #     if line_acc > best_line_acc:
        #     #         best_line_acc = line_acc
        #     #         best_char_acc = char_acc
        #     #         best_iters = total_iters
        #     #         torch.save(netC_T.state_dict(), 'checkpoints/' + opt.name + '/' + 'best_CT.pth')
        #     #         torch.save(netC_S.state_dict(), 'checkpoints/' + opt.name + '/' + 'best_CS.pth')
        #     #         torch.save(netG_T.state_dict(), 'checkpoints/' + opt.name + '/' + 'best_GT.pth')
        #     #         torch.save(netG_S.state_dict(), 'checkpoints/' + opt.name + '/' + 'best_GS.pth')
        #     #         torch.save(netD_T.state_dict(), 'checkpoints/' + opt.name + '/' + 'best_DT.pth')
        #     #         torch.save(netD_S.state_dict(), 'checkpoints/' + opt.name + '/' + 'best_DS.pth')
        #     #         print('save best models in ', 'checkpoints/'+opt.name)
        #     #
        #     #     visual_items = [total_iters, line_acc, char_acc, best_line_acc, best_char_acc, best_iters]
        #     #     visualizer.print_test_losses(visual_items)
        #     #
        #     #     with open(os.path.join(opt.checkpoints_dir, opt.name, 'state_file.txt'), 'w') as f:
        #     #         f.write("latest_iter: %d\n" % total_iters)
        #     #         f.write("line_acc: %f\n" % line_acc)
        #     #         f.write("char_acc: %f\n" % char_acc)
        #     #         f.write("best_line_acc: %f\n" % best_line_acc)
        #     #         f.write("best_char_acc: %f\n" % best_char_acc)
        #     #         f.write("best_iter: %d\n" % best_iters)
        #
        #     iter_data_time = time.time()
        #
        # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        # model.update_learning_rate()                     # update learning rates at the end of every epoch.
