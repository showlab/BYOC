import os
import glob
import torch
import cv2
import json
import numpy as np
from scipy.io import loadmat
from data.base_dataset import BaseDataset


class Exp2BsDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--exp_basis_folder', type=str)
        parser.add_argument('--blendshape_file', type=str)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.sample_list = sorted(glob.glob(os.path.join(opt.exp_basis_folder, '*.mat')))
        with open(opt.blendshape_file, 'r') as j:
            blendshape_dict = json.load(j)
            self.blendshape_dict = {}
            for k, v in blendshape_dict.items():
                self.blendshape_dict[os.path.basename(k)] = v

    def __getitem__(self, index):
        exp_file = self.sample_list[index]
        exp_array = loadmat(exp_file)['exp'][0]
        exp_tsr = torch.from_numpy(exp_array)

        img_path = exp_file.replace('mat', 'png')
        img_tsr = self.load_img(img_path)

        blendshape_weight = self.blendshape_dict[os.path.basename(exp_file).replace('mat', 'jpg')]
        blendshape_weight = np.asarray([float(x) for x in blendshape_weight])
        blendshape_tsr = torch.from_numpy(blendshape_weight)

        return {'exp': exp_tsr,
                'blendshapes': blendshape_tsr,
                'images': img_tsr,
                'img_paths': img_path}

    def load_img(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # crop img
        height, width = img.shape[:2]
        min_size = min(height, width)
        img = img[:min_size, :min_size]

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
        return len(self.sample_list)
