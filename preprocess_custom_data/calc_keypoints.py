import cv2
import json
import numpy as np
def readjson(file_name):
    with open(file_name,'r') as f:
        x = json.load(f)
    return x
def drawLms(img, lms, color=(0, 0, 255),name = "lms.png"):
    img = img.copy()
    for lm in lms:
        cv2.circle(img, tuple(lm), 10, color, 1)
    cv2.imwrite(name,img)
    # cv2.imshow("1",img)
    # cv2.waitKey()
    return img
def test_keypoint():
    j1=readjson("/home/algo/yangxinhang/NeuralHaircut/implicit-hair-data/data/monocular/person_0/openpose_kp/img_0000_keypoints.json")
    img =cv2.imread("/home/algo/yangxinhang/NeuralHaircut/implicit-hair-data/data/monocular/person_0/image/img_0000.png")
    lms = np.array(j1['people'][0]['face_keypoints_2d']).reshape((-1,3))[:,:2].astype('int')
    drawLms(img,lms,name="x1.png")
from openpose import pyopenpose as op

# 配置 OpenPose 参数
params = {
    "model_folder": "path/to/openpose/models",  # OpenPose 模型文件夹路径
    "face": True,  # 启用面部关键点检测
    "hand": False,  # 禁用手部关键点检测
    "disable_blending": True,  # 禁用融合（加速运行）
}

# 初始化 OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 读取输入图像
image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)

# 运行 OpenPose
datum = op.Datum()
datum.cvInputData = image
opWrapper.emplaceAndPop([datum])

# 获取人体姿势关键点
pose_keypoints = datum.poseKeypoints

# 获取面部关键点
face_keypoints = datum.faceKeypoints

# 打印结果（每个关键点包含 (x, y, score)）
print("Pose keypoints:", pose_keypoints)
print("Face keypoints:", face_keypoints)

# 绘制结果（可选）
cv2.imshow("OpenPose Result", datum.cvOutputData)
cv2.waitKey(0)
cv2.destroyAllWindows()
