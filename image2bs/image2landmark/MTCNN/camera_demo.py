import cv2
from mtcnn import FaceDetector
from PIL import Image
import numpy

detector = FaceDetector()


def camera_detect():
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()

        # 将 OpenCV 格式的图片转换为 PIL.Image
        pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 绘制带人脸框的标注图
        drawed_pil_im = detector.draw_bboxes(pil_im)
        # 再转回 OpenCV 格式用于视频显示
        frame = cv2.cvtColor(numpy.asarray(drawed_pil_im), cv2.COLOR_RGB2BGR)

        cv2.imshow("Face Detection", frame)
        # 输入 q 的时候结束循环（退出检测程序）
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_detect()
