import os
import glob
import shutil

# from MTCNN.mtcnn import FaceDetector
from PIL import Image
from tqdm import tqdm

from image2landmark.MTCNN.mtcnn import FaceDetector


# save_draw_img = False
def image2landmark():
    detector = FaceDetector()

    img_folder = 'test_images'
    img_list = glob.glob(os.path.join(img_folder, '*.jpg')) + glob.glob(os.path.join(img_folder, '*.png'))

    if os.path.exists(os.path.join(img_folder, 'detections')):
        shutil.rmtree(os.path.join(os.path.join(img_folder, 'detections')))

    lmk_save_folder = os.path.join(img_folder, 'detections')
    os.makedirs(lmk_save_folder, exist_ok=True)

    # if save_draw_img:
    #     img_save_folder = os.path.join('datasets/M111_combined_result')
    #     os.makedirs(img_save_folder, exist_ok=True)
    # print(len(img_list))

    for img_path in tqdm(img_list):
        # print(img_path)
        try:
            # landmark：[右眼x, 左眼x, 鼻子x, 右嘴角x, 左嘴角x, 右眼y, 左眼y, 鼻子y, 右嘴角y, 左嘴角y]
            image = Image.open(img_path)
            bboxes, landmarks = detector.detect(image)
            landmarks = landmarks[0]

            img_name = os.path.basename(img_path)
            text_path = os.path.join(lmk_save_folder, img_name.replace('jpg', 'txt'))

            text_file = open(text_path, "w")
            text_file.write("{:.2f}\t{:.2f}\n".format(landmarks[1], landmarks[1 + 5]))  # 左眼
            text_file.write("{:.2f}\t{:.2f}\n".format(landmarks[0], landmarks[0 + 5]))  # 右眼
            text_file.write("{:.2f}\t{:.2f}\n".format(landmarks[2], landmarks[2 + 5]))  # 鼻子
            text_file.write("{:.2f}\t{:.2f}\n".format(landmarks[4], landmarks[4 + 5]))  # 左嘴角
            text_file.write("{:.2f}\t{:.2f}\n".format(landmarks[3], landmarks[3 + 5]))  # 右嘴角
            # print("File successfully written: {}".format(text_path))
            text_file.close()

            # 绘制并保存标注图
            # if save_draw_img:
            #     draw_path = os.path.join(img_save_folder, img_name)
            #     drawed_image = detector.draw_bboxes(image)
            #     drawed_image.save(draw_path)
        except Exception as e:
            print(e)
            continue
