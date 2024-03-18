import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_face2face
from data.HDFS.hdfs_path_lib import load as hload


class VirtualHumanModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        networks_face2face.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.loss_names = ['D', 'G', 'GAN', 'G_feat', 'VGG']
        self.visual_names = ['real_img', 'fake_img']
        if self.isTrain:
            self.model_names = ['G', 'E', 'D']
        else:  # during test time, only load G
            self.model_names = ['G', 'E']

        # define networks
        self.netG = networks_face2face.define_G(opt)
        self.netE = networks_face2face.define_E(opt)

        if self.isTrain:  # define discriminators
            self.netD = networks_face2face.define_D(opt)

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks_face2face.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt).to(self.device)  # define GAN loss.
            self.criterionGFeat = torch.nn.L1Loss().to(self.device)
            self.criterionVGG = networks_face2face.VGGLoss(self.opt).to(self.device)

            # TTUR training schema
            if opt.no_TTUR:
                beta1, beta2 = opt.beta1, opt.beta2
                G_lr, D_lr = opt.lr, opt.lr
            else:
                beta1, beta2 = 0, 0.9
                G_lr, D_lr = opt.lr / 2, opt.lr * 2
            self.old_lr = opt.lr

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(),
                                                                self.netE.parameters()), lr=G_lr, betas=(beta1, beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=D_lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def load_network(self, network, save_path, net_label):
        try:
            network.load_state_dict(hload(save_path))
        except:
            pretrained_dict = hload(save_path)
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
        self.real_img = input['images'].to(self.device)
        self.blendshape = input['blendshapes'].to(self.device)
        self.image_paths = input['img_paths']

    def forward(self):
        style = [self.netE(self.blendshape)]
        self.fake_img, _ = self.netG(style)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        # Combined loss and calculate gradients
        loss_D = sum([loss_D_real, loss_D_fake]).mean() * self.opt.lambda_D
        loss_D.backward()
        return loss_D

    def backward_D(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_img)
        self.loss_D = self.backward_D_basic(self.netD, self.real_img, fake_B)

    def backward_G(self):
        self.loss_GAN = self.criterionGAN(self.netD(self.fake_img), True, for_discriminator=False)
        self.loss_G_feat = self.compute_GAN_Feat_loss(self.netD(self.fake_img), self.netD(self.real_img))
        self.loss_VGG = self.criterionVGG(self.fake_img, self.real_img) * self.opt.lambda_vgg

        loss_G_list = [self.loss_GAN, self.loss_G_feat, self.loss_VGG]

        self.loss_G = sum(loss_G_list).mean()
        self.loss_G.backward()

    def compute_GAN_Feat_loss(self, pred_fake, pred_real):
        num_D = len(pred_fake)
        GAN_Feat_loss = self.FloatTensor(1).fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionGFeat(pred_fake[i][j], pred_real[i][j].detach())
                if j == 0:
                    unweighted_loss *= self.opt.lambda_recon
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
        return GAN_Feat_loss

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G and E_motion
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D
        self.optimizer_D.step()  # update D_A and D_B's weights

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

