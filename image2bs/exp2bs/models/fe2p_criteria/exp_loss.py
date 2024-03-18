import torch
from torch import nn
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class ExpLoss(nn.Module):
    def __init__(self, opt):
        super(ExpLoss, self).__init__()
        print('Loading FER model')
        self.model = VGG('VGG19')
        checkpoint = torch.load(opt.FER_path)
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def rgb2gray(self, x):
        return x[:, 0, :, :] * 0.299 + x[:, 1, :, :] * 0.587 + x[:, 2, :, :] * 0.114

    def pre_process(self, x):
        x = self.rgb2gray(x)
        x = torch.unsqueeze(x, dim=1)
        x = F.interpolate(x, size=(48, 48))
        x = x.repeat(repeats=[1, 3, 1, 1])
        return x

    def extract_feats(self, x):
        x = self.pre_process(x)
        x_feats = self.model(x)
        return x_feats

    def forward(self, real_img, fake_img):
        real_feat = self.extract_feats(real_img)
        fake_feat = self.extract_feats(fake_img)
        loss = F.l1_loss(fake_feat, real_feat)

        return loss

