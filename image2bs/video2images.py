"""
将视频转换为图片，可以为多个文件夹下的图片。
注：在程序使用前需先配置好main中的地址
视频路径：video_path_list = [path1, path2, ...](路径数量可以为[1,n]，每个路径下的视频数也可为[1,m])
    path1                path2             ....
     |------video1.avi      |-----video1.avi
     |------vidoe2.avi      |-----...
     |------....
图片存储路径：image_save_dir = save_path(存储方式则将按以下方式）
    save_path
     | -------path1_name
                |----video1
                        |----jpg1.jpg
                        |----jpg2,jpg
                |----video2
                ...
     |-------path2_name
     ...
"""

import cv2
import os
from pathlib import Path

VID_FORMATS = ('.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv', '.mp3')


def videos2images(root_video_path, root_save_dir):
    for video_dir_path in root_video_path:
        # 1.检测读取文件路径是否正确
        path_video = Path(video_dir_path)
        if path_video.is_dir():
            print(video_dir_path + ' to images')
            videos = os.listdir(video_dir_path)
        else:
            print('\033[31mLine36 error: \033[31m' + video_dir_path + 'is not exist!')
            return

        # 2. 生成存储文件夹
        # save_name_dir = Path(path_video.name)
        # save_name_dir = os.path.join(root_save_dir, save_name_dir)
        # if not os.path.exists(save_name_dir):
        #     os.makedirs(save_name_dir)
        save_name_dir = root_save_dir

        file_count = 0
        for video in videos:
            # 判断是否为视频文件,如果不是视频文件则跳过并进行说明
            if Path(video).suffix in VID_FORMATS:
                file_count += 1  # 视频文件数+1
                # save_jpg_dir = os.path.join(save_name_dir, Path(video).stem)
                save_jpg_dir = save_name_dir
                if not os.path.exists(save_jpg_dir):
                    os.makedirs(save_jpg_dir)
                each_video_path = os.path.join(path_video, video)
                save_dir = save_jpg_dir
            else:
                print('\033[33mLine56 warning: \033[33m' + os.path.basename(video) + ' is not a video file, so skip.')
                continue

            # 3. 开始转换。打印正在处理文件的序号和他的文件名，并开始转换
            # print('\033[38m' + str(file_count) + ':' + Path(video).stem + '\033[38m')
            cap = cv2.VideoCapture(each_video_path)

            flag = cap.isOpened()
            if not flag:
                print("\033[31mLine 65 error\033[31m: open" + each_video_path + "error!")

            frame_count = 0  # 给每一帧标号
            while True:
                frame_count += 1
                flag, frame = cap.read()
                if not flag:  # 如果已经读取到最后一帧则退出
                    break

                zero_count = 5 - len(str(frame_count))
                zero_name = ""
                for i in range(zero_count):
                    zero_name += "0"

                if os.path.exists(
                        save_dir + zero_name + str(frame_count) + '.jpg'):  # 在源视频不变的情况下，如果已经创建，则跳过
                    break
                cv2.imwrite(save_dir + '\\' + zero_name + str(frame_count) + '.jpg', frame)

            cap.release()
            print('\033[38m' + Path(video).stem + ' has saved to ' + save_dir + '. \033[38m')  # 表示一个视频片段已经转换完成


def video2images():
    video_path_list_ = [r'./test_video']
    image_save_dir_ = r'./test_images'
    image_list = os.listdir(image_save_dir_)
    if image_list:
        for root, dirs, files in os.walk(image_save_dir_):
            for file in files:
                os.remove(os.path.join(root, file))

    # if image_list:
    #     for image in image_list:
    #         os.remove(os.path.join(image_save_dir_, image))

    videos2images(video_path_list_, image_save_dir_)


if __name__ == '__main__':
    # 需要转换的视频路径列表，直达视频文件(自定义修改）
    video_path_list = [r'./test_video']

    # 预期存储在的主文件夹，即'result'文件夹下
    image_save_dir = r'./test_images'
    # path_save = Path(image_save_dir)
    # if not path_save.exists():
    #     path_save.mkdir()
    # 进行转换
    videos2images(video_path_list, image_save_dir)
