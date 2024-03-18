import os
import glob
import torch
import cv2
import numpy as np
from data.base_dataset import BaseDataset


class MixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--character_id', type=str, default='M111')
        parser.add_argument('--dataroot_real', type=str)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.character_data_path = os.path.join(opt.dataroot, opt.character_id)
        parts = ['composition', 'individual']
        self.image_paths = []
        self.blendshape_dict = {}

        for p in parts:
            # get img paths
            img_folder = os.path.join(self.character_data_path, 'camera_front_' + p)
            image_paths = glob.glob(os.path.join(img_folder, '*.jpg'))
            self.image_paths = self.image_paths + sorted(image_paths)[1:]

            # get blendshape weights
            with open(os.path.join(self.character_data_path, '{}_{}.csv'.format(opt.character_id, p)), 'r') as f:
                for line in f:
                    line_data = line.strip().split(',')
                    defined_name = line_data[0]
                    weight = [float(x) for x in line_data[1:]]

                    id = defined_name.split('_')[0]
                    assert id == opt.character_id
                    num = int(defined_name[:-4].split('_')[1]) + 1
                    new_name = os.path.join(img_folder, '{}_{}.jpg'.format(id, num))
                    assert new_name in self.image_paths

                    self.blendshape_dict[new_name] = np.asarray(weight, dtype=np.float32)

        self.img_dir_real = opt.dataroot_real
        self.image_paths_real = glob.glob(os.path.join(self.img_dir_real, '*.jpg'))

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img_tsr = self.load_img(img_path, crop=True,
                                crop_hori=75, crop_top=0, crop_down=150)

        img_path_real = self.image_paths_real[index % len(self.image_paths_real)]
        img_tsr_real = self.load_img(img_path_real, crop=False)

        blendshape_weight = self.blendshape_dict[img_path]
        blendshape_tsr = torch.from_numpy(blendshape_weight)

        return {'blendshapes': blendshape_tsr,
                'images': img_tsr,
                'images_real': img_tsr_real,
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
