import os

from scipy.io import loadmat
import math
import csv
from tqdm import tqdm


def read_pose_angle():
    csv_data = []
    mat_relative_path = "./landmark2exp/checkpoints/facerecon/results/test_images/epoch_20_000000"
    all_files = os.listdir(mat_relative_path)
    # csv_row = []
    for i in tqdm(all_files):
        if i[-3:] == "mat":
            # path = r"./mat/00001.mat"  # mat文件路径
            data = loadmat(mat_relative_path + '/' + i)  # 读取mat文件
            pose_angle = data['angle']
            pose_angle_list = list(pose_angle[0])
            # print(pose_angle_list)

            # print(pose_angle[0])
            pitch = pose_angle_list[0] * 180 / math.pi
            yaw = -pose_angle_list[1] * 180 / math.pi
            roll = -pose_angle_list[2] * 180 / math.pi
            # pitch = pose_angle_list[0]
            # yaw = pose_angle_list[1]
            # roll = pose_angle_list[2]

            csv_row = [i, pitch, yaw, roll]
            csv_data.append(csv_row)
    res = "./results/predicted_pose_angle.csv"

    with open(res, 'w') as file:
        writer = csv.writer(file)
        # Use writerows() not writerow()
        writer.writerows(csv_data)


if __name__ == '__main__':
    read_pose_angle()
