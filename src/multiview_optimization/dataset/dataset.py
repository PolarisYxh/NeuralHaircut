from torch.utils.data import Dataset, DataLoader
import face_alignment
import numpy as np
from skimage.io import imread
import os
import torch
import cv2 as cv
import sys
import json
from .openpose_data import OpenposeData
import pickle
from .cameras import OptimizableCameras
from Util.get_face_info import get_face_info,angle2matrix
import cv2
def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict
def drawLms(img, lms, color=(0, 255, 0),name = "1"):
    img1 = img.copy()
    for lm in lms:
        cv2.circle(img1, tuple(lm), 2, color, 1)
    cv2.imwrite(name+".png",img1)
    # cv2.imshow(name,img1)
    # cv2.waitKey()
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    
    return intrinsics, pose

class Multiview_dataset(Dataset):
    def __init__(self, image_path='', scale_path='', camera_path='' , openpose_kp_path='', pixie_init_path='', fitted_camera_path='', views_idx='',  device='cuda', batch_size=1):
        self.device = device

        self.cams = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)),camera_path))
        self.scale_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),scale_path)
        self.image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),image_path)
        self.openpose_kp_path = None
        if openpose_kp_path!="":
            self.openpose_kp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),openpose_kp_path)
            print(self.openpose_kp_path)
        self.pixie_init_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),pixie_init_path)
        self.batch_size = batch_size
        
        self.fitted_camera_path = fitted_camera_path

        imgs_list = sorted(os.listdir(self.image_path))
        imgs_list_full = sorted(os.listdir(self.image_path))
        
        if views_idx:
            with open(views_idx, 'rb') as f:
                filter_idx = pickle.load(f) 
                imgs_list =  [imgs_list[i] for i in filter_idx]

        images_np = np.stack([imread(os.path.join(self.image_path, im_name)) for im_name in imgs_list])
        images = torch.from_numpy((images_np.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)).float()

        #  camera
        if self.scale_path:
            scale_mat = np.eye(4, dtype=np.float32)        
            with open(self.scale_path, 'rb') as f:
                transform = pickle.load(f)
                print('upload transform', transform, self.scale_path)
                scale_mat[:3, :3] *= transform['scale']
                scale_mat[:3, 3] = np.array(transform['translation'])

            world_mats_np = [self.cams['arr_0'][idx].astype(np.float32) for idx in range(len(imgs_list_full) )]
            scale_mats_np = [scale_mat.astype(np.float32) for idx in range(len(imgs_list_full) )]
        else:
            world_mats_np = [self.cams['world_mat_%d' % idx].astype(np.float32) for idx in range(len(imgs_list_full) )]
            scale_mats_np = [self.cams['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(imgs_list_full) )]
        # self.insight_face_info = get_face_info("Util"),False)
        intrinsics_all = []
        pose_all = []
        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())

        intrinsics_all = torch.stack(intrinsics_all).to(device)   # [n_images, 4, 4]
        pose_all = torch.stack(pose_all).to(device)  # [n_images, 4, 4]

        if views_idx:
            with open(views_idx, 'rb') as f:
                filter_idx = pickle.load(f) 
                
                pose_all =  pose_all[filter_idx]
                intrinsics_all = intrinsics_all[filter_idx]
        else:
            filter_idx = np.arange(len(imgs_list))
        

        self.camera_model = None
        if fitted_camera_path:
            self.camera_model = OptimizableCameras(len(imgs_list), pretrain_path=fitted_camera_path)
            with torch.no_grad():
                intrinsics_all, pose_all = self.camera_model(torch.arange(len(imgs_list)), intrinsics_all, pose_all)

        intrinsics_all = intrinsics_all[:, :3, :3]
#         landmarks
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        
        lmks = [self.fa.get_landmarks_from_image(images_np[i])[0]  if self.fa.get_landmarks_from_image(images_np[i]) else None for i in range(len(imgs_list))]
        fa3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
        lmks3d = [fa3d.get_landmarks_from_image(images_np[i])[0]  if fa3d.get_landmarks_from_image(images_np[i]) else None for i in range(len(imgs_list))]  

        # took views that have openpose keypoints
        # self.good_views  = []
        # mapping = dict(zip(filter_idx, np.arange(len(imgs_list))))
        # unmapping = dict(zip(np.arange(len(imgs_list)), filter_idx))
        
        # for i in range(len(imgs_list)):
        #     # faces, frames, framesForHair = self.insight_face_info.get_faces(images_np[i])
        #     # if faces is not None:
        #     #     self.good_views.append(mapping[i]) 
        #     if lmks[i] is not None:
        #         drawLms(images_np[i],lmks[i].astype('int'),name=f"{i}")
        self.good_views = list(range(0,len(imgs_list_full)))
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)),image_path,"../good_view.json")):
            self.good_views  = readjson(os.path.join(os.path.dirname(os.path.dirname(__file__)),image_path,"../good_view.json"))['good_view']
        self.openpose_data = None
        if isinstance(self.openpose_kp_path,str):
            data_openpose = []
            self.good_views = []
            sns = sorted(os.listdir(self.openpose_kp_path))
            for sn in sns:
                with open(os.path.join(self.openpose_kp_path, sn), 'r') as f:
                    data_openpose.append(json.load(f))

            mapping = dict(zip(filter_idx, np.arange(len(imgs_list))))
            unmapping = dict(zip(np.arange(len(imgs_list)), filter_idx))


            for i in range(len(data_openpose)):
                if i in filter_idx:
                    if len(data_openpose[i]['people'])>0:
                        if sum(data_openpose[i]['people'][0]['face_keypoints_2d']) > 0  and sum(data_openpose[i]['people'][0]['pose_keypoints_2d'])>0    :#
                            self.good_views.append(mapping[i])     #[0, 9, 20, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, ...]  32 

            self.openpose_data = OpenposeData(path=self.openpose_kp_path, views=self.good_views, device=self.device, filter_views_mapping=unmapping)    
        
        self.good_views = [i for i in self.good_views if lmks[i] is not None] # For some views otained landmarks could be bad
        print(f"filter good view:{self.good_views}")
        self.num_views = len(self.good_views)
#         self.good_views = [0, 9, 27, 28, 30, 31, 33, 34, 35, 63] # person_1

        self.nimages = min(len(self.good_views), self.batch_size)

        # self.good_views = np.array(self.good_views)[::len(self.good_views)//self.nimages][:self.nimages]
        self.good_views = np.array(self.good_views)[:self.nimages]

        self.lmks = torch.from_numpy(np.stack([lmks[i] for i in self.good_views]))
        self.lmks3d = torch.from_numpy(np.stack([lmks3d[i] for i in self.good_views]))
        self.images = images[self.good_views]
        self.poses = torch.inverse(pose_all[self.good_views])
        self.intrinsics_all = intrinsics_all[self.good_views]
    
    def get_filter_views(self):
        return torch.tensor(self.good_views).to(self.device)
        
    def __getitem__(self, index):
        print(index)
        if self.openpose_data ==None:
            return {
                'img': self.images[index].to(self.device), 
                'lmks': self.lmks[index].to(self.device),
                'lmks3d':self.lmks3d[index].to(self.device), 
                'extrinsics_rvec': self.poses[index, :3, :3].to(self.device),
                'extrinsics_tvec': self.poses[index, :3, 3].to(self.device),
                'frame_ids': torch.tensor(index, dtype=torch.long).to(self.device),
                'intrinsics': self.intrinsics_all[index].to(self.device),
                # 'openpose_lmks': self.openpose_data.get_sample(index)
               } 
        else:
            return {
                    'img': self.images[index].to(self.device), 
                    'lmks': self.lmks[index].to(self.device),
                    'lmks3d':self.lmks3d[index].to(self.device), 
                    'extrinsics_rvec': self.poses[index, :3, :3].to(self.device),
                    'extrinsics_tvec': self.poses[index, :3, 3].to(self.device),
                    'frame_ids': torch.tensor(index, dtype=torch.long).to(self.device),
                    'intrinsics': self.intrinsics_all[index].to(self.device),
                    'openpose_lmks': self.openpose_data.get_sample(index)
                } 
    
    def __len__(self):
        return self.nimages 