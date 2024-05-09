import cv2
import os
from Util.get_face_info import get_face_info,angle2matrix
import json
import numpy as np
def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)
# 读取视频文件
case_name = "person_5"
work_dir = os.path.join(os.path.dirname(__file__),f"../implicit-hair-data/data/monocular/{case_name}")
os.makedirs(os.path.join(work_dir,'video_frames'),exist_ok=True)
video_capture = cv2.VideoCapture(os.path.join(work_dir,f"person_5.mp4"))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

insight_face_info = get_face_info("Util/",False)
# 视频帧计数器
frame_count = 0
count = -1
step = total_frames//80
# 逐帧读取视频并保存为图像
good_view = []
while True:
    success, frame = video_capture.read()
    count+=1
    if count%step!=0:
        continue
    if not success:
        break
    faces = insight_face_info.get_origin_faces(frame)
    if len(faces)>=1:
        face_id=0
        face = faces[face_id]
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = int((bbox[2] + bbox[0]) / 2), int((bbox[3] + bbox[1]) / 2)
        if frame.shape[0]>frame.shape[1]:
            if center[1]>frame.shape[1]//2 and center[1]+frame.shape[1]//2<frame.shape[0]:
                frame = frame[center[1]-frame.shape[1]//2:center[1]+frame.shape[1]//2]
            elif center[1]<=frame.shape[1]//2:
                frame = frame[0:frame.shape[1]]
            elif center[1]+frame.shape[1]//2>=frame.shape[0]:
                frame = frame[:-frame.shape[1]]
        else:
            if center[0]>frame.shape[0]//2 and center[0]+frame.shape[0]//2<frame.shape[1]:
                frame = frame[:,center[0]-frame.shape[0]//2:center[0]+frame.shape[0]//2]
            elif center[0]<=frame.shape[0]//2:
                frame = frame[:,0:frame.shape[0]]
            elif center[0]+frame.shape[0]//2>=frame.shape[1]:
                frame = frame[:-frame.shape[0]]
        # cv2.imwrite(os.path.join(work_dir,'video_frames/img_{:04d}.png').format(frame_count), frame)
        if np.max(np.abs(face['pose']))<40:
            good_view.append(frame_count)
    else:
        #裁剪成1：1的照片
        m_l = min(frame.shape)
        if frame.shape[0]>frame.shape[1]:
            frame = frame[(frame.shape[0]-frame.shape[1])//4:-(frame.shape[0]-frame.shape[1])//4*3]
        else:
            frame = frame[:,(frame.shape[1]-frame.shape[0])//2:-(frame.shape[1]-frame.shape[0])//2]
    # 保存图像文件
        # cv2.imwrite(os.path.join(work_dir,'video_frames/img_{:04d}.png').format(frame_count), frame)
    
    # 增加帧计数器
    frame_count += 1
writejson(os.path.join(work_dir,'good_view.json'),{'good_view':good_view})
# 关闭视频捕获对象
video_capture.release()
