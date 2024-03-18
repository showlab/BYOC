import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_face2face
from models.networks_face2face.architecture import VGGFace19
from data.HDFS.hdfs_path_lib import load as hload
import torch.nn.functional as F
from models.networks_face2face.encoders.helpers import l2_norm
from models.networks_face2face.encoders.model_irse import Backbone as IRSE_Backbone


class Face2FaceV6Model(BaseModel):
    """
    optimize the face region crop by hard crop
    specifically designed for official vox2 crop
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--use_face_region_recon', type=int, default=0)
        parser.add_argument('--id_cls', type=int, default=0)
        parser.add_argument('--feat_loss_type', type=str, default='l1')
        parser.add_argument('--use_pose_cycle_loss', type=int, default=0)
        networks_face2face.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.loss_names = ['D', 'G', 'GAN', 'G_feat', 'VGG', 'ArcFaceId']
        if opt.vgg_face:
            self.loss_names.append('VGGFace')
        if opt.use_face_region_recon:
            self.loss_names.append('crop_recon')
        if opt.id_cls:
            self.loss_names.append('IdCls')
        if opt.use_pose_cycle_loss:
            self.loss_names.append('PoseCyc')
        self.visual_names = ['real_A', 'real_B', 'aug_B', 'fake_B',
                             'cropped_A', 'cropped_B', 'cropped_fake_B']
        if self.isTrain:
            self.model_names = ['G', 'D', 'E_idt', 'E_motion']
        else:  # during test time, only load G
            self.model_names = ['G', 'E_idt', 'E_motion']

        # define networks
        self.netG = networks_face2face.define_G(opt)
        self.netE_idt = networks_face2face.define_V(opt)
        self.netE_motion = networks_face2face.define_E(opt)

        if self.isTrain:  # define discriminators
            self.netD = networks_face2face.define_D(opt)
            self.load_network(self.netE_idt, opt.identity_encoder_pretrain_path, 'E_idt (netV)')
            self.netE_motion.load_pretrain()

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks_face2face.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt).to(self.device)  # define GAN loss.
            self.criterionVGG = networks_face2face.VGGLoss(self.opt).to(self.device)
            self.criterionCLS = torch.nn.CrossEntropyLoss().to(self.device)

            if str(self.opt.feat_loss_type).lower() == 'l1':
                self.criterionGFeat = torch.nn.L1Loss().to(self.device)
            elif str(self.opt.feat_loss_type).lower() in ['l2', 'mse']:
                self.criterionGFeat = torch.nn.MSELoss().to(self.device)        # changed from L1 to MSE (L2)
            else:
                raise RuntimeError('Unsupported GAN feat loss type')

            self.netArc = IRSE_Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
            self.netArc.load_state_dict(hload('hdfs://haruna/home/byte_labcv_default/user/baizechen/face2face_ckpt/model_ir_se50.pth'))
            self.netArc.to(self.device)
            self.netArc.eval()
            self.set_requires_grad(self.netArc, False)

            if opt.vgg_face:
                self.VGGFace = VGGFace19(self.opt)
                self.criterionVGGFace = networks_face2face.VGGLoss(self.opt, self.VGGFace).to(self.device)

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
                                                                self.netE_motion.parameters(),
                                                                self.netE_idt.parameters()), lr=G_lr, betas=(beta1, beta2))
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
        self.real_A = input['realA'].to(self.device)
        self.real_B = input['realB'].to(self.device)
        self.aug_B = input['augB'].to(self.device)
        self.id_targets = input['id_targets'].to(self.device)
        self.image_paths_A = input['A_paths']
        self.image_paths_B = input['B_paths']
        self.image_paths = input['A_paths']

    def cosin_metric(self, x1, x2):
        return (1 - torch.sum(x1 * x2, dim=1)).mean()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        id_feature, self.id_scores = self.netE_idt(self.real_A)
        self.motion_feature = self.netE_motion.forward_feature(self.aug_B)
        style = [torch.cat([id_feature[0], self.motion_feature], 1)]
        self.fake_B, _ = self.netG(style)
        with torch.no_grad():
            _, self.fake_id_scores = self.netE_idt(self.fake_B)

        if self.isTrain:
            self.cropped_A = F.interpolate(self.real_A[:, :, 20:164, 40:-40], size=(112, 112), mode='bilinear', align_corners=True)
            self.cropped_B = F.interpolate(self.real_B[:, :, 20:164, 40:-40], size=(112, 112), mode='bilinear', align_corners=True)
            self.cropped_fake_B = F.interpolate(self.fake_B[:, :, 20:164, 40:-40], size=(112, 112), mode='bilinear', align_corners=True)

        if self.opt.use_pose_cycle_loss:
            with torch.no_grad():
                self.fake_motion_feature = self.netE_motion.forward_feature(self.fake_B)

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
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B)

    def backward_G(self):
        self.loss_GAN = self.criterionGAN(self.netD(self.fake_B), True, for_discriminator=False)
        self.loss_G_feat = self.compute_GAN_Feat_loss(self.netD(self.fake_B), self.netD(self.real_B))
        self.loss_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_vgg
        self.loss_ArcFaceId = self.cosin_metric(self.netArc(self.cropped_A), self.netArc(self.cropped_fake_B)) * self.opt.lambda_id_dist

        loss_G_list = [self.loss_GAN, self.loss_G_feat, self.loss_VGG, self.loss_ArcFaceId]

        if self.opt.use_face_region_recon:
            self.loss_crop_recon = self.criterionGFeat(self.cropped_fake_B, self.cropped_B) * self.opt.lambda_recon * self.opt.lambda_feat
            loss_G_list.append(self.loss_crop_recon)

        if self.opt.vgg_face:
            self.loss_VGGFace = self.criterionVGGFace(self.fake_B, self.real_B, layer=2) * self.opt.lambda_vggface
            loss_G_list.append(self.loss_VGGFace)

        if self.opt.id_cls:
            self.loss_IdCls = self.criterionCLS(self.id_scores, self.id_targets) + \
                              self.criterionCLS(self.fake_id_scores, self.id_targets)
            loss_G_list.append(self.loss_IdCls)

        if self.opt.use_pose_cycle_loss:
            self.loss_PoseCyc = self.cosin_metric(l2_norm(self.motion_feature), l2_norm(self.fake_motion_feature)) * self.opt.lambda_pose_cyc
            loss_G_list.append(self.loss_PoseCyc)

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
