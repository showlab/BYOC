import torch
import itertools
from .base_model import BaseModel
from . import networks_face2face
from models.fe2p_criteria.exp_loss import ExpLoss
from models.fe2p_criteria.seg_loss import SegLoss
import torch.nn.functional as F


class FE2PModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--BiSeNet_path', type=str, default='')
        parser.add_argument('--FER_path', type=str, default='')
        parser.add_argument('--netE_path', type=str, default='')
        parser.add_argument('--netG_path', type=str, default='')
        parser.add_argument('--lambda_Exp', type=float, default=0.5)
        parser.add_argument('--lambda_Seg', type=float, default=0.5)
        parser.add_argument('--lambda_Loop', type=float, default=1.0)
        parser.add_argument('--norm_blendshape', type=str, default='sigmoid')
        networks_face2face.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.loss_names = ['Exp', 'Seg', 'Loop']
        self.visual_names = ['real_img', 'fake_img']
        self.model_names = ['V']

        # define networks
        self.netV = networks_face2face.define_V(opt)

        self.netE = networks_face2face.define_E(opt)
        self.netG = networks_face2face.define_G(opt)
        self.load_network(self.netE, self.opt.netE_path, 'netE')
        self.load_network(self.netG, self.opt.netG_path, 'netG')
        self.netE.eval()
        self.netG.eval()
        self.set_requires_grad(self.netE, False)
        self.set_requires_grad(self.netG, False)

        if self.isTrain:
            self.criterionLoop = torch.nn.L1Loss().to(self.device)
            self.criterionExp = ExpLoss(self.opt).to(self.device)
            self.criterionSeg = SegLoss(self.opt).to(self.device)

            # TTUR training schema
            if opt.no_TTUR:
                beta1, beta2 = opt.beta1, opt.beta2
                G_lr, D_lr = opt.lr, opt.lr
            else:
                beta1, beta2 = 0, 0.9
                G_lr, D_lr = opt.lr / 2, opt.lr * 2
            self.old_lr = opt.lr

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netV.parameters()), lr=G_lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_G)

    def load_network(self, network, save_path, net_label):
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            pretrained_dict = torch.load(save_path)
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                network.load_state_dict(pretrained_dict)
                if self.opt.verbose:
                    print('Pretrained network {} has excessive layers; Only loading layers that are used'.format(net_label))
            except:
                print('Pretrained network {} has fewer layers; The following are not initialized:'.format(net_label))
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v
                not_initialized = set()
                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])
                print(sorted(not_initialized))
                network.load_state_dict(model_dict)

    def set_input(self, input):
        self.real_img = input['img'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        self.blendshape = self.netV.forward_feature(self.real_img)
        style = self.netE(self.blendshape)
        self.fake_img, _ = self.netG([style])

        self.fake_img = F.interpolate(self.fake_img[:, :, 10:-20, 10:-10], size=(self.opt.crop_size, self.opt.crop_size))

        self.fake_blendshape = self.netV.forward_feature(self.fake_img)

    def backward_G(self):
        self.loss_Exp = self.criterionExp(self.real_img, self.fake_img) * self.opt.lambda_Exp
        self.loss_Seg = self.criterionSeg(self.real_img, self.fake_img) * self.opt.lambda_Seg
        self.loss_Loop = self.criterionLoop(self.blendshape, self.fake_blendshape) * self.opt.lambda_Loop

        loss_G_list = [self.loss_Exp, self.loss_Seg, self.loss_Loop]

        self.loss_G = sum(loss_G_list).mean()
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()      # compute fake images and reconstruction images.
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

    def update_learning_rate(self, epoch):
        if epoch > self.opt.n_epochs:
            lrd = self.opt.lr / self.opt.n_epochs_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

