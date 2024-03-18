import os
import glob
import torch
import cv2
import json
import numpy as np
from scipy.io import loadmat
from data.base_dataset import BaseDataset


class Exp2BsMixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--exp_basis_folder_virtual', type=str)
        parser.add_argument('--exp_basis_folder_real', type=str)
        parser.add_argument('--blendshape_file', type=str)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.sample_list_virtual = sorted(glob.glob(os.path.join(opt.exp_basis_folder_virtual, '*.mat')))
        self.sample_list_real = sorted(glob.glob(os.path.join(opt.exp_basis_folder_real, '*.mat')))

        with open(opt.blendshape_file, 'r') as j:
            blendshape_dict = json.load(j)
            self.blendshape_dict = {}
            for k, v in blendshape_dict.items():
                k = k[-15:]
                self.blendshape_dict[os.path.basename(k)] = v
                print(k)

    def __getitem__(self, index):
        exp_file = self.sample_list_virtual[index]
        exp_array = loadmat(exp_file)['exp'][0]
        exp_tsr_virtual = torch.from_numpy(exp_array)

        img_path = exp_file.replace('mat', 'png')
        img_tsr_virtual = self.load_img(img_path)

        # blendshape_weight = self.blendshape_dict[os.path.basename(exp_file).replace('mat', 'jpg')]
        # print("Look Here: " + exp_file)
        # blendshape_weight = self.blendshape_dict[(exp_file[-15:]).replace('mat', 'jpg')]
        blendshape_weight = self.blendshape_dict[(exp_file[-9:]).replace('mat', 'png')]


        blendshape_weight = np.asarray([float(x) for x in blendshape_weight])
        blendshape_tsr = torch.from_numpy(blendshape_weight)

        exp_file = self.sample_list_real[index]
        exp_array = loadmat(exp_file)['exp'][0]
        exp_tsr_real = torch.from_numpy(exp_array)

        img_path = exp_file.replace('mat', 'png')
        img_tsr_real = self.load_img(img_path)

        return {'exp_virtual': exp_tsr_virtual,
                'images_virtual': img_tsr_virtual,
                'blendshapes': blendshape_tsr,
                'exp_real': exp_tsr_real,
                'images_real': img_tsr_real,
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
        return len(self.sample_list_virtual)
