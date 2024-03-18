from models.networks_face2face.encoders.model_irse import Backbone
import torch
id_path = 'model_ir_se50.pth'
from torch.nn import functional as F


def extract_id(id_net, x):
    x_temp = x[:, :, :245, :]
    h, w = x_temp.shape[2], x_temp.shape[3]
    y_r = F.interpolate(x_temp, size=(88, 92), mode='bilinear', align_corners=True)
    zeros = torch.zeros((x.shape[0], x.shape[1], 112, 112)).to(x.device)
    zeros[:, :, 112-88:, 10:102] = y_r
    x_feats = id_net(zeros)
    return x_feats


id_net = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
id_net.load_state_dict(torch.load(id_path, map_location='cpu'), strict=True)

