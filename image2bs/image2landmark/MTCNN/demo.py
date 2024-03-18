from mtcnn import FaceDetector
from PIL import Image

# 人脸检测对象。优先使用GPU进行计算（会自动判断GPU是否可用）
# 你也可以通过设置 FaceDetector("cpu") 或者 FaceDetector("cuda") 手动指定计算设备
detector = FaceDetector()

image = Image.open("./images/image.jpg")

# 检测人脸，返回人脸位置坐标
# 其中bboxes是一个n*5的列表、landmarks是一个n*10的列表，n表示检测出来的人脸个数，数据详细情况如下：
# bbox：[左上角x坐标, 左上角y坐标, 右下角x坐标, 右下角y坐标, 检测评分]
# landmark：[右眼x, 左眼x, 鼻子x, 右嘴角x, 左嘴角x, 右眼y, 左眼y, 鼻子y, 右嘴角y, 左嘴角y]
bboxes, landmarks = detector.detect(image)

# 绘制并保存标注图
drawed_image = detector.draw_bboxes(image)
drawed_image.save("./images/drawed_image.jpg")

# 裁剪人脸图片并保存
face_img_list = detector.crop_faces(image, size=64)
for i in range(len(face_img_list)):
    face_img_list[i].save("./images/face_" + str(i + 1) + ".jpg")
