import os
import glob
import torch
import cv2
from data.base_dataset import BaseDataset


class FE2PDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_dir = opt.dataroot
        self.image_paths = glob.glob(os.path.join(self.img_dir, '*.jpg'))

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img_tsr = self.load_img(img_path)

        return {'img': img_tsr, 'path': img_path}

    def load_img(self, image_path, M=None):
        img = cv2.imread(image_path)

        if img is None:
            raise Exception('None Image')

        if M is not None:
            img = cv2.warpAffine(img, M, (self.opt.crop_size, self.opt.crop_size), borderMode=cv2.BORDER_REPLICATE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.to_Tensor(img)
        return img

    def to_Tensor(self, img):
        if img.ndim == 3:
            wrapped_img = img.transpose(2, 0, 1) / 255.0
        elif img.ndim == 4:
            wrapped_img = img.transpose(0, 3, 1, 2) / 255.0
        else:
            wrapped_img = img / 255.0
        wrapped_img = torch.from_numpy(wrapped_img).float()

        return wrapped_img * 2 - 1

    def __len__(self):
        return len(self.image_paths)
