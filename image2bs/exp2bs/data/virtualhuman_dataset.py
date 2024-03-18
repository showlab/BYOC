import os
import torch
import numpy as np
import cv2
import json
import multiprocessing as mp
import dataloader
from data.base_dataset import BaseDataset
from data.HDFS.hdfs_path_lib import hopen


def get_keys(args):
    return dataloader.KVReader(*args).list_keys()


class VirtualHumanDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--hdfs_path', type=str, default='')
        parser.add_argument('--blendshape_path', type=str, default='')
        parser.add_argument('--set0_blendshape_list', type=list, default=['roll', 'tx', 'ty', 'tz'])
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.hdfs_path = opt.hdfs_path
        self.set0_blendshape_list = opt.set0_blendshape_list
        print('The following blendshapes will be set to 0.\n {}'.format(str(self.set0_blendshape_list)))
        with mp.Pool(1) as p:
            self.all_keys = p.map(get_keys, [(opt.hdfs_path, 32)])[0]
        with hopen(self.opt.blendshape_path, 'r') as f:
            self.blendshapes = json.loads(f.read().decode('utf-8'))
        # self.blendshape_order = ['Afraid', 'Angry', 'Bereft', 'Bored', 'BrowDown_L', 'BrowDown_R', 'BrowInnerUp',
        #                          'BrowOuterUp_L', 'BrowOuterUp_R', 'CheekPuff', 'CheeksSquint_L', 'CheeksSquint_R',
        #                          'Concentrate', 'Confident', 'Confused', 'Contempt', 'Desire', 'Disgust', 'Drunk',
        #                          'Excitement', 'EyeBlink_L', 'EyeBlink_R', 'EyeSquint_L', 'EyeSquint_R', 'EyeWide_L',
        #                          'EyeWide_R', 'Fear', 'Fierce', 'Flirting', 'Frown', 'Glare', 'Happy', 'Ignore',
        #                          'Incredulous', 'Irritated', 'JawForward', 'JawLeft', 'JawOpen', 'JawRight', 'LIll',
        #                          'LookDown', 'LookLeft', 'LookRight', 'LookUp', 'MouthDimple_L', 'MouthDimple_R',
        #                          'MouthFrown_L', 'MouthFrown_R', 'MouthFunnel', 'MouthLeft', 'MouthLowerDown_L',
        #                          'MouthLowerDown_R', 'MouthPress_L', 'MouthPress_R', 'MouthPucker', 'MouthRight',
        #                          'MouthRollLower', 'MouthRollUpper', 'MouthShrug_Lower', 'MouthShrug_Upper',
        #                          'MouthSmile_L', 'MouthSmile_R', 'MouthStretch_L', 'MouthStretch_R', 'MouthUpperUp_L',
        #                          'MouthUpperUp_R', 'NoseSneer_L', 'NoseSneer_R', 'Pain', 'Pleased', 'Pouty', 'Rage',
        #                          'Sad', 'Sarcastic', 'Scream', 'Serious', 'Shock', 'Silly', 'SmileFullFace',
        #                          'SmileOpenFullFace', 'Snarl', 'Surprised', 'Suspicious', 'Tired', 'Triumph', 'Wink',
        #                          'nromal', 'vAA', 'vEE', 'vEH', 'vER', 'vF', 'vIH', 'vIY', 'vK', 'vL', 'vM', 'vOW',
        #                          'vS', 'vSH', 'vT', 'vTH', 'vUW', 'vW', 'tx', 'ty', 'tz', 'heading', 'pitch', 'roll']
        self.blendshape_order = ['BrowDown_L', 'BrowDown_R', 'BrowInnerUp', 'BrowOuterUp_L', 'BrowOuterUp_R',
                                 'CheekPuff', 'CheeksSquint_L', 'CheeksSquint_R', 'EyeBlink_L', 'EyeBlink_R',
                                 'EyeSquint_L', 'EyeSquint_R', 'EyeWide_L', 'EyeWide_R', 'JawForward', 'JawLeft',
                                 'JawOpen', 'JawRight', 'LookDown', 'LookLeft', 'LookRight', 'LookUp', 'MouthDimple_L',
                                 'MouthDimple_R', 'MouthFrown_L', 'MouthFrown_R', 'MouthFunnel', 'MouthLeft',
                                 'MouthLowerDown_L', 'MouthLowerDown_R', 'MouthPress_L', 'MouthPress_R', 'MouthPucker',
                                 'MouthRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrug_Lower',
                                 'MouthShrug_Upper', 'MouthSmile_L', 'MouthSmile_R', 'MouthStretch_L', 'MouthStretch_R',
                                 'MouthUpperUp_L', 'MouthUpperUp_R', 'NoseSneer_L', 'NoseSneer_R',
                                 'heading', 'pitch', 'roll']
        print('{} keys originally'.format(len(self.all_keys)))
        # self.filter_blendshapes()
        print('{} keys after filtering'.format(len(self.all_keys)))
        print('Init Virtual Human Render HDFS dataset!')
        self.size = len(self.all_keys)

    def filter_blendshapes(self):
        new_keys = []
        for key in self.all_keys:
            flag = True
            blendshape = self.blendshapes[key]
            for k, v in blendshape.items():
                if k not in self.blendshape_order + ['tx', 'ty', 'tz'] and v > 0.2:
                    flag = False
                    break
            if flag:
                new_keys.append(key)
        self.all_keys = new_keys

    def __getitem__(self, index):
        img_keys = [self.all_keys[i % self.size] for i in index]    # make sure index is within then range
        img_strs = self.reader.read_many(img_keys)

        imgs, blendshapes = [], []
        for key, img_str in zip(img_keys, img_strs):
            imgs.append(self.to_Tensor(self.load_img(img_str)))
            blendshapes.append(self.dict2list(self.blendshapes[key]))

        imgs = torch.stack(imgs, dim=0)
        blendshapes = torch.from_numpy(np.asarray(blendshapes, dtype=np.float32))

        return {'images': imgs, 'blendshapes': blendshapes, 'img_paths': img_keys}

    def load_img(self, img_str):
        img_array = np.frombuffer(img_str, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception('None Image')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(self.opt.crop_size, self.opt.crop_size))
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

    def dict2list(self, d):
        l = []
        for k in self.blendshape_order:
            if k in self.set0_blendshape_list:
                l.append(0.0)
            elif k not in d.keys():
                l.append(0.0)
            else:
                l.append(d[k])
        return l

    def __len__(self):
        return self.size

    def name(self):
        return 'VirtualHumanRenderDatasetHDFS'

