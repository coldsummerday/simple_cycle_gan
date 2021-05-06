from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=10, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        # parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--n_class', type=int, default=6129, help='the number of classes')
        parser.add_argument('--chn_dict_path', type=str, default='./list/chinese_charset.txt',
                                            help='the datalist containing img path and labels of source dataset')
        parser.add_argument('--source_list', type=str, default='./list/src_generate.txt', help='the datalist containing img path and labels of source dataset')
        parser.add_argument('--target_list', type=str, default='./list/train_1w.txt', help='the datalist containing img path and labels of target dataset')
        parser.add_argument('--test_list', type=str, default='./list/test_without_keys.txt', help='the datalist containing img path and labels of test dataset')
        parser.add_argument('--pretrain_ctc_model', type=str, default='./checkpoints/train_srcg/ocr-44000.pth', help='the pretrained ctc model')
        parser.add_argument('--test_freq', type=int, default=5000, help='frequency of testing model')
        parser.add_argument('--restore', type=int, default=0, help='restore the trained model')


        parser.add_argument('--pad_h_ratio', type=str, default='0.05,0.15', help='padding height ratio')
        parser.add_argument('--pad_w_ratio', type=str, default='0.05,0.30', help='padding width ratio')
        parser.add_argument('--font_obl_a', type=str, default='-5,5', help='rotate angle')
        parser.add_argument('--font_shade_range', type=str, default='180,255', help='controlling shade of font')
        parser.add_argument('--font_clarity_range', type=str, default='0.5,1', help='controlling clarity of font')
        parser.add_argument('--font_size_range', type=str, default='25,50', help='control width of font')


        ##use lmdb dataset
        parser.add_argument('--source_lmdb', type=str, default='./lmdbdataset/src_generate/',
                            help='the lmdb file containing img  and labels of source dataset')
        parser.add_argument('--target_lmdb', type=str, default='./lmdbdataset/train_1w_zhb/',
                            help='the lmdb file containing img  and labels of target dataset')
        parser.add_argument('--test_lmdb', type=str, default='./lmdbdataset/val_1w_zhb',
                            help='the lmdb file containing img  and labels of test dataset')
        parser.add_argument("--charset",type=str,default="./data/chinese_charset.txt",
                            help="the char set for  code  each char one line")
        self.isTrain = True
        return parser
