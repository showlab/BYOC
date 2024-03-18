import os
import glob
import torch
import cv2
import json
import numpy as np
from data.base_dataset import BaseDataset


class VirtualHumanOUDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--img_folder', type=str)
        parser.add_argument('--blendshape_file', type=str)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.image_paths = glob.glob(os.path.join(opt.img_folder, '*.jpg'))
        with open(opt.blendshape_file, 'r') as j:
            blendshape_dict = json.load(j)
            self.blendshape_dict = {}
            for k, v in blendshape_dict.items():
                self.blendshape_dict[os.path.basename(k)] = v

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img_tsr = self.load_img(img_path, crop=True, crop_hori=75, crop_top=0, crop_down=150)

        blendshape_weight = self.blendshape_dict[os.path.basename(img_path)]
        blendshape_weight = np.asarray([float(x) for x in blendshape_weight])
        blendshape_tsr = torch.from_numpy(blendshape_weight).float()

        return {'blendshapes': blendshape_tsr,
                'images': img_tsr,
                'img_paths': img_path}

    def load_img(self, image_path, M=None, crop=False, crop_hori=100, crop_top=100, crop_down=100):
        img = cv2.imread(image_path)

        if img is None:
            raise Exception('None Image')

        if M is not None:
            img = cv2.warpAffine(img, M, (self.opt.crop_size, self.opt.crop_size), borderMode=cv2.BORDER_REPLICATE)

        if crop:
            img = img[crop_top:-crop_down, crop_hori:-crop_hori]
            img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))

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
