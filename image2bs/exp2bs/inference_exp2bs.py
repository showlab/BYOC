import os
# from options.face2face_options.test_options import TestOptions
import torch

# from exp2bs.options.test_options import TestOptions
# from exp2bs.models import create_model
# import util.util as util
import numpy as np
import glob
import cv2
from tqdm import tqdm
from scipy.io import loadmat
import csv

from .models import create_model
from .options.test_options import TestOptions

np.random.seed(123)


def load_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = to_Tensor(img)
    return img


def to_Tensor(img):
    if img.ndim == 3:
        wrapped_img = img.transpose(2, 0, 1) / 255.0
    elif img.ndim == 4:
        wrapped_img = img.transpose(0, 3, 1, 2) / 255.0
    else:
        wrapped_img = img / 255.0
    wrapped_img = torch.from_numpy(wrapped_img).float()

    return wrapped_img * 2 - 1


def inference_random_sample(opt, model, sample_num=22):
    sample_list_real = sorted(glob.glob(os.path.join(opt.exp_basis_folder_real, '*.mat')))
    total_num = len(sample_list_real)
    # indices = np.random.randint(0, total_num - 1, size=sample_num)

    # f = open(os.path.join('../results/predicted_blendshape.csv'), 'w')
    f = open('results/predicted_blendshape.csv', 'w')

    # for index in tqdm(indices):
    for index in tqdm(range(len(sample_list_real))):
        exp_file = sample_list_real[index]
        exp_array = loadmat(exp_file)['exp'][0]
        exp_tsr_real = torch.from_numpy(exp_array).unsqueeze(dim=0).cuda()

        blendshape_tsr = torch.randn(size=(1, 50)).float().cuda()

        img_path = exp_file.replace('mat', 'png')
        img_tsr_real = load_img(img_path).unsqueeze(dim=0).cuda()

        data = {'exp_virtual': exp_tsr_real,
                'images_virtual': img_tsr_real,
                'blendshapes': blendshape_tsr,
                'exp_real': exp_tsr_real,
                'images_real': img_tsr_real,
                'img_paths': img_path}

        model.set_input(data)

        # save image file
        # model.forward_visuals()
        # real_img = model.real_img[0]
        # img_size = real_img.size(-2)
        #
        # real_img = real_img[:, :, :img_size]
        # fake_real_img = F.interpolate(model.fake_real_img, size=(img_size, img_size))[0]
        #
        # img_list = [util.tensor2im_v2(real_img, tile=False),
        #             util.tensor2im_v2(fake_real_img, tile=False)]
        # big_img = np.concatenate(img_list, axis=1)
        # save_name = 'results/Exp2BsExamples/{}'.format(os.path.basename(img_path))
        # util.save_image_v2(big_img, save_name, create_dir=True)

        # save pred blendshape file
        model.forward()
        writer = csv.writer(f)
        value_weight_str = ["%.6f" % x for x in model.blendshape_pred[0].detach().cpu().numpy().tolist()]
        cur_row = [os.path.basename(img_path)] + value_weight_str
        writer.writerow(cur_row)

    return


def main():
    torch.manual_seed(0)
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.isTrain = False

    # # opt = TestOptions().parse()
    # opt.dataset_mode = "exp2bsmix"
    # # opt.exp_basis_folder_virtual = "../landmark2exp/checkpoints/facerecon/results/M111_combined_cam_front/epoch_20_000000"
    # opt.exp_basis_folder_real = "../landmark2exp/checkpoints/facerecon/results/test_images/epoch_20_000000"
    # # opt.blendshape_file = "Deep3DFaceRecon_pytorch/datasets/M111_combined_blendshape.json"
    # opt.model = "exp2bsmix"
    # opt.netV = "exp2bsleakyclamp"
    # opt.netE = "fc50blendshape"
    # opt.netG = "modulate"
    # opt.checkpoints_dir = "exp2bs/checkpoint/EXP2BSMIX"
    # opt.crop_size = 256
    # opt.name = "Exp2BlendshapeMixRegress"
    # opt.style_dim = 512
    # opt.norm_blendshape = None

    # print(opt.checkpoints_dir)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()

    inference_random_sample(opt, model)


def rename_example(gan_folder, maya_folder, dst_folder):
    with open('results/predicted_blendshape_123.csv', 'r') as f:
        for idx, line in enumerate(f):
            img_name = line.strip().split(',')[0]

            gan_img_path = os.path.join(gan_folder, img_name)
            gan_img = cv2.imread(gan_img_path)

            crop_hori, crop_top, crop_down = 75, 0, 150
            maya_img_path = os.path.join(maya_folder, 'M111_{}.jpg'.format(idx + 1))
            maya_img = cv2.imread(maya_img_path)
            maya_img = maya_img[crop_top:-crop_down, crop_hori:-crop_hori]
            maya_img = cv2.resize(maya_img, (224, 224))

            dst_img = np.concatenate([gan_img, maya_img], axis=1)
            cv2.imwrite(os.path.join(dst_folder, img_name), dst_img)


def rename_example_for_poster_M(gan_folder, maya_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    with open('results/predicted_blendshape_123.csv', 'r') as f:
        for idx, line in enumerate(f):
            img_name = line.strip().split(',')[0]

            gan_img_path = os.path.join(gan_folder, img_name)
            gan_img = cv2.imread(gan_img_path)
            gan_img = gan_img[:, :224]

            crop_hori, crop_top, crop_down = 75, 0, 150
            maya_img_path = os.path.join(maya_folder, 'M109_{}.jpg'.format(idx + 1))
            maya_img = cv2.imread(maya_img_path)
            maya_img = maya_img[crop_top:-crop_down, crop_hori:-crop_hori]
            maya_img = cv2.resize(maya_img, (224, 224))

            dst_img = np.concatenate([gan_img, maya_img], axis=1)
            cv2.imwrite(os.path.join(dst_folder, img_name), dst_img)


def rename_example_for_poster_F(gan_folder, maya_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    with open('results/predicted_blendshape_123.csv', 'r') as f:
        for idx, line in enumerate(f):
            img_name = line.strip().split(',')[0]

            gan_img_path = os.path.join(gan_folder, img_name)
            gan_img = cv2.imread(gan_img_path)
            gan_img = gan_img[:, :224]

            crop_hori, crop_top, crop_down = 75, 50, 100
            maya_img_path = os.path.join(maya_folder, 'F141_{}.jpg'.format(idx + 1))
            maya_img = cv2.imread(maya_img_path)
            maya_img = maya_img[crop_top:-crop_down, crop_hori:-crop_hori]
            maya_img = cv2.resize(maya_img, (224, 224))

            dst_img = np.concatenate([gan_img, maya_img], axis=1)
            cv2.imwrite(os.path.join(dst_folder, img_name), dst_img)


def inference_exp2bs():
    main()
    # main(opt)
