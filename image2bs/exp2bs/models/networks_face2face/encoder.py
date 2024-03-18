import torch.nn as nn
from exp2bs.models.networks_face2face.base_network import BaseNetwork
from exp2bs.util import util
import torch
import torch.nn.functional as F
from exp2bs.models.networks_face2face.FAN_feature_extractor import FAN_use
from torchvision.models.vgg import vgg19_bn
from exp2bs.models.networks_face2face.vision_network import ResNeXt50
from exp2bs.models.networks_face2face.encoders.model_irse import Backbone as IRSE_Backbone
from exp2bs.models.networks_face2face.encoders.helpers import l2_norm


class ResNeXtEncoder(ResNeXt50):
    def __init__(self, opt):
        super(ResNeXtEncoder, self).__init__(opt)


class VGGEncoder(BaseNetwork):
    def __init__(self, opt):
        super(VGGEncoder, self).__init__()
        self.model = vgg19_bn(num_classes=opt.num_classes)

    def forward(self, x):
        return self.model(x)


class IRSEEncoder(BaseNetwork):
    def __init__(self, opt):
        super(IRSEEncoder, self).__init__()
        self.opt = opt
        self.model = IRSE_Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')

        if opt.id_embedding_dim != 512:
            self.embedding = nn.Linear(512, self.opt.id_embedding_dim)
            print('increase the id embedding dim to {}'.format(opt.id_embedding_dim))
            if self.opt.use_id_auto_enc:
                self.recon = nn.Linear(self.opt.id_embedding_dim, 512)
                print('use auto-encoder loss to further constrain the embedding up-dim')

    def forward(self, x):
        with torch.no_grad():
            id_feature = self.model(x)
        return id_feature

    def forward_with_embedding(self, x):
        with torch.no_grad():
            id_feature = self.model(x)

        if self.opt.id_embedding_dim != 512:
            id_embedding = self.embedding(id_feature)
            if self.opt.use_id_auto_enc:
                recon_id_feature = self.recon(id_embedding)
                return id_feature, id_embedding, l2_norm(recon_id_feature)
            else:
                return id_feature, id_embedding
        else:
            return id_feature

    def load_pretrain(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.opt.identity_encoder_pretrain_path
        check_point = torch.load(ckpt_path)
        print("=> loading checkpoint '{}'".format(ckpt_path))
        self.model.load_state_dict(check_point)


class FanEncoder(BaseNetwork):
    def __init__(self, opt):
        super(FanEncoder, self).__init__()
        self.opt = opt
        pose_dim = self.opt.pose_dim
        self.model = FAN_use(opt)
        self.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, opt.num_classes))

        # mapper to mouth subspace
        self.to_mouth = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        self.mouth_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, 512 - pose_dim))
        self.mouth_fc = nn.Sequential(nn.ReLU(), nn.Linear(512 * opt.clip_len, opt.num_classes))

        # mapper to head pose subspace
        self.to_headpose = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        self.headpose_embed = nn.Sequential(nn.ReLU(), nn.Linear(512, pose_dim))
        self.headpose_fc = nn.Sequential(nn.ReLU(), nn.Linear(pose_dim * opt.clip_len, opt.num_classes))

    def load_pretrain(self):
        check_point = torch.load(self.opt.motion_encoder_pretrain_path)
        print("=> loading checkpoint '{}'".format(self.opt.motion_encoder_pretrain_path))
        util.copy_state_dict(check_point, self.model, strip='model.')

    def forward_feature(self, x):
        net = self.model(x)
        if str(self.opt.norm_blendshape).lower() == 'sigmoid':
            # print('using sigmoid')
            net = torch.sigmoid(net)
        elif str(self.opt.norm_blendshape).lower() == 'clip':
            # print('using clamp')
            net = torch.clamp(net, min=0.0, max=1.0)
        return net

    def forward(self, x):
        x0 = x.view(-1, self.opt.output_nc, self.opt.crop_size, self.opt.crop_size)
        net = self.forward_feature(x0)
        scores = self.classifier(net.view(-1, self.opt.num_clips, 512).mean(1))
        return net, scores


class Fcv1Encoder(BaseNetwork):
    def __init__(self, opt):
        super(Fcv1Encoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(110, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class Fcv2Encoder(BaseNetwork):
    def __init__(self, opt):
        super(Fcv2Encoder, self).__init__()
        self.opt = opt
        self.fc_blendshape = nn.Linear(104, 384)
        self.fc_cam = nn.Linear(6, 128)

    def forward(self, x):
        x1 = self.fc_blendshape(x[:, :104])
        x2 = self.fc_cam(x[:, 104:])
        out = torch.cat([x1, x2], dim=1)

        return out


class Fcv3Encoder(BaseNetwork):
    def __init__(self, opt):
        super(Fcv3Encoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(110, 256)
        self.fc2 = nn.Linear(256, 512)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)

        return x


class Fc46blendshapeEncoder(BaseNetwork):
    def __init__(self, opt):
        super(Fc46blendshapeEncoder, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(46, 512)

    def forward(self, x):
        out = self.fc(x)

        return out


class Fc46blendshapeV2Encoder(BaseNetwork):
    def __init__(self, opt):
        super(Fc46blendshapeV2Encoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(46, 256)
        self.fc2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        out = self.fc2(x)
        return out


class FFHQ3DMMFCV1Encoder(BaseNetwork):
    def __init__(self, opt):
        super(FFHQ3DMMFCV1Encoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(132, 256)
        self.fc2 = nn.Linear(256, 512)
        self.relu = nn.ReLU()

    def forward(self, x_exp, x_id, x_pose):
        x = torch.cat([x_exp, x_id, x_pose], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class FFHQ3DMMFCV2Encoder(BaseNetwork):
    def __init__(self, opt):
        super(FFHQ3DMMFCV2Encoder, self).__init__()
        self.opt = opt
        self.fc_exp = nn.Linear(51, 128)
        self.fc_id = nn.Linear(75, 256)
        self.fc_pose = nn.Linear(6, 128)

    def forward(self, x_exp, x_id, x_pose):
        x1 = self.fc_exp(x_exp)
        x2 = self.fc_id(x_id)
        x3 = self.fc_pose(x_pose)
        out = torch.cat([x1, x2, x3], dim=1)

        return out


class Fc50blendshapeEncoder(BaseNetwork):
    def __init__(self, opt):
        super(Fc50blendshapeEncoder, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(50, 512)

    def forward(self, x):
        out = self.fc(x)

        return out


class Fc50blendshapeV2Encoder(BaseNetwork):
    def __init__(self, opt):
        super(Fc50blendshapeV2Encoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(50, 256)
        self.fc2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        out = self.fc2(x)
        return out


class Exp2BsEncoder(BaseNetwork):
    def __init__(self, opt):
        super(Exp2BsEncoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(64, 384)
        self.fc2 = nn.Linear(384, 50)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        out = self.fc2(x)

        if str(self.opt.norm_blendshape).lower() == 'sigmoid':
            out = torch.sigmoid(out)
        elif str(self.opt.norm_blendshape).lower() == 'clip':
            out = torch.clamp(out, min=0.0, max=1.0)

        return out


class Exp2BsOneLayerEncoder(BaseNetwork):
    def __init__(self, opt):
        super(Exp2BsOneLayerEncoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(64, 50)

    def forward(self, x):
        x = self.fc1(x)

        return x


class Exp2BsLeakyEncoder(BaseNetwork):
    def __init__(self, opt):
        super(Exp2BsLeakyEncoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 50)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        out = self.fc2(x)

        return out


class Exp2BsClampEncoder(BaseNetwork):
    def __init__(self, opt):
        super(Exp2BsClampEncoder, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 50)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        out = self.fc2(x)
        out = torch.clamp(out, min=0.0, max=1.0)

        return out


# Backbone
class Exp2BsLeakyClampEncoder(BaseNetwork):
    def __init__(self, opt):
        super(Exp2BsLeakyClampEncoder, self).__init__()
        self.opt = opt
        # origin
        self.fc1 = nn.Linear(64, 256)
        # 50， 25， 66， 113
        self.fc2 = nn.Linear(256, 50)
        # self.bn1 = nn.BatchNorm1d(256)

        # 113 - updated
        # self.fc1 = nn.Linear(64, 256)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.fc2 = nn.Linear(256, 512)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.fc3 = nn.Linear(512, 256)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.fc4 = nn.Linear(256, 113)

    def forward(self, x):
        # origin
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        # x = F.relu(x)
        out = self.fc2(x)
        out = torch.clamp(out, min=0.0, max=1.0)

        # # 113 - updated
        # x = self.fc1(x)
        # x = self.bn1(x)
        # # x = F.relu(x)
        # x = F.leaky_relu(x, negative_slope=0.1)
        # x = self.fc2(x)
        # x = self.bn2(x)
        # # x = F.relu(x)
        # x = F.leaky_relu(x, negative_slope=0.1)
        # x = self.fc3(x)
        # x = self.bn3(x)
        # # x = F.relu(x)
        # x = F.leaky_relu(x, negative_slope=0.1)
        # out = self.fc4(x)
        # out = torch.clamp(out, min=0.0, max=1.0)

        return out
