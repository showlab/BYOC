import face_alignment
import os
import cv2
import skimage.transform as trans
import argparse
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_affine(src):
    dst = np.array([[87,  59],
                    [137,  59],
                    [112, 120]], dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    return M


def affine_align_img(img, M, crop_size=224):
    warped = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
    return warped


def affine_align_3landmarks(landmarks, M):
    new_landmarks = np.concatenate([landmarks, np.ones((3, 1))], 1)
    affined_landmarks = np.matmul(new_landmarks, M.transpose())
    return affined_landmarks


def get_eyes_mouths(landmark):
    three_points = np.zeros((3, 2))
    three_points[0] = landmark[36:42].mean(0)
    three_points[1] = landmark[42:48].mean(0)
    three_points[2] = landmark[60:68].mean(0)
    return three_points


def get_mouth_bias(three_points):
    bias = np.array([112, 120]) - three_points[2]
    return bias


def align_folder(folder_path, folder_save_path):

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
    preds = fa.get_landmarks_from_directory(folder_path)

    for img_pth in preds.keys():
        pred_points = np.array(preds[img_pth])
        if pred_points is None or len(pred_points.shape) != 3:
            print('preprocessing failed 1')
            return False
        else:
            num_faces, size, _ = pred_points.shape
            if num_faces == 1 and size == 68:
                three_points = get_eyes_mouths(pred_points[0])
                M = get_affine(three_points)
                affined_3landmarks = affine_align_3landmarks(three_points, M)
                bias = get_mouth_bias(affined_3landmarks)
                M_i = M.copy()
                M_i[:, 2] = M[:, 2] + bias
                img = cv2.imread(img_pth)
                print(img.shape)
                wrapped = affine_align_img(img, M_i)
                img_save_path = os.path.join(folder_save_path, os.path.basename(img_pth))
                cv2.imwrite(img_save_path, wrapped)
            else:
                print(img_pth, 'preprocessing failed 2')
                return False
    print('cropped files saved at {}'.format(folder_save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder_path', help='the folder which needs processing')
    parser.add_argument('--output_folder_path', help='the folder that store the processed results')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)

    align_folder(args.input_folder_path, args.output_folder_path)


if __name__ == '__main__':
    main()
