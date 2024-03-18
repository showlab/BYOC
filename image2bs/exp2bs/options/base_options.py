import sys
import argparse
import os
# from exp2bs.util import util
import torch
# import models
# import data
import pickle

from exp2bs import models, data
from exp2bs.util import util


# from exp2bs import models, data
# from exp2bs import util


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # added from cyclegan option or as needed
        parser.add_argument('--dataroot', default='./', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--identity_encoder_pretrain_path', type=str, default='', help='identity encoder pretrain path')
        parser.add_argument('--motion_encoder_pretrain_path', type=str, default='', help='motion encoder pretrain path')
        parser.add_argument('--idt_cls_pretrain_path', type=str, default='', help='identity classification encoder pretrain path')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')

        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        # experiment specifics
        parser.add_argument('--name', type=str, default='Exp2BlendshapeMixRegress', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        parser.add_argument('--num_classes', type=int, default=5830, help='num classes')
        parser.add_argument('--checkpoints_dir', type=str, default='exp2bs/checkpoint/EXP2BSMIX', help='models are saved here')
        parser.add_argument('--model', type=str, default='exp2bsmix', help='which model to use, rotate|rotatespade')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='resize_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--crop_len', type=int, default=16, help='Crop len')
        parser.add_argument('--target_crop_len', type=int, default=0, help='Crop len')
        parser.add_argument('--crop', action='store_true', help='whether to crop the image')
        parser.add_argument('--clip_len', type=int, default=1, help='num of imgs to process')
        parser.add_argument('--pose_dim', type=int, default=12, help='num of imgs to process')
        parser.add_argument('--num_clips', type=int, default=1, help='num of clips to process')
        parser.add_argument('--num_inputs', type=int, default=1, help='num of inputs to the network')
        parser.add_argument('--feature_encoded_dim', type=int, default=2560, help='dim of reduced id feature')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--generate_interval', type=int, default=1, help='select frames to generate')
        parser.add_argument('--style_feature_loss', action='store_true', help='style_feature_loss')

        # for setting inputsf
        parser.add_argument('--dataset_mode', type=str, default='exp2bsmix')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--verbose', action='store_true', help='just add')

        # for generator
        parser.add_argument('--netG', type=str, default='modulate', help='selects model to use for netG (modulate)')
        parser.add_argument('--netV', type=str, default='exp2bsleakyclamp', help='selects model to use for netV (mobile | id)')
        parser.add_argument('--netE', type=str, default='fc50blendshape', help='selects model to use for netV (mobile | fan)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image|projection)')
        parser.add_argument('--D_input', type=str, default='single', help='(concat|single|hinge)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        parser.add_argument('--style_dim', type=int, default=512, help='# of encoder filters in the first conv layer')
        parser.add_argument('--motion_dim', type=int, default=512, help='dim of the motion embedding')
        parser.add_argument('--id_embedding_dim', type=int, default=512, help='dim of the id embedding')

        ####################### weight settings ###################################################################
        parser.add_argument('--vgg_face', action='store_true', help='if specified, use VGG face feature matching loss')
        parser.add_argument('--VGGFace_pretrain_path', type=str, default='', help='VGGFace pretrain path')
        parser.add_argument('--lambda_vggface', type=float, default=5.0, help='weight for vggface loss')

        parser.add_argument('--lambda_recon', type=float, default=1.0, help='weight for image reconstruction')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_D', type=float, default=1.0, help='D loss weight')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_id_dist', type=float, default=2.0, help='weight for id dist loss')
        parser.add_argument('--lambda_pose_cyc', type=float, default=25.0, help='weight for pose cycle consistency')

        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--optimize_E_in_D', action='store_true', help='whether optimize E_motion params in D optimizer')

        ############################## optimizer #############################
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

        # continue train
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_modes = opt.dataset_mode.split(',')

        if len(dataset_modes) == 1:
            dataset_option_setter = data.get_option_setter(dataset_mode)
            parser = dataset_option_setter(parser, self.isTrain)
        else:
            for dm in dataset_modes:
                dataset_option_setter = data.get_option_setter(dm)
                parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # lt options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()

        # added here for option conflict between cyclgan and PC-AVS
        opt.num_threads = opt.nThreads
        opt.batch_size = opt.batchSize
        opt.preprocess = opt.preprocess_mode

        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)
        # Set semantic_nc based on the option.
        # This will be convenient in many places
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
