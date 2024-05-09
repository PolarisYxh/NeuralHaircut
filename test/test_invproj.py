import trimesh
import numpy as np
import os
import pickle
import cv2
from NeuS.models.dataset import load_K_Rt_from_P
def test_colmap_intrinsic(path_to_scene):
    """测试colmap读取cameras.bin和images.bin得到的透视投影内外参是否正确
    """    
    # idx = readjson("/home/algo/yangxinhang/NeuralHaircut/preprocess_custom_data/front.json")["back"]
    # pc = np.array(trimesh.load(os.path.join(path_to_scene, 'head_prior_wo_eyes.obj'),process=False).vertices)
    pc = np.array(trimesh.load(os.path.join(path_to_scene, 'point_cloud_cropped_normalize.ply'),process=False).vertices)
    # pc=pc[idx,:]
    from glob import glob
    camera_dict = np.load(os.path.join(path_to_scene, "cameras.npz"))
    
    # Define scale into unit sphere
    scale_mat = np.eye(4, dtype=np.float32)
    with open(os.path.join(path_to_scene,"scale.pickle"), 'rb') as f:
        trans = pickle.load(f)
        # print('upload transform', transform, conf['path_to_scale'])
        scale_mat[:3, :3] *= trans['scale']
        scale_mat[:3, 3] = np.array(trans['translation'])
    
    images_file = f'{path_to_scene}/colmap/sparse_txt/images.txt'
    points_file = f'{path_to_scene}/colmap/sparse_txt/points3D.txt'
    camera_file = f'{path_to_scene}/colmap/sparse_txt/cameras.txt'
    
    # Parse colmap cameras and used images
    intrinsic_matrixs = []
    with open(camera_file) as f:
        lines = f.readlines()
        for line in lines[3:]:
            u = float(line.split()[4])
            h, w = [int(x) for x in lines[3].split()[5: 7]]

            intrinsic_matrix = np.array([
                [u, 0, h, 0],
                [0, u, w, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            intrinsic_matrixs.append(intrinsic_matrix)

    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    with open(images_file) as f:
        images_file_lines = f.readlines()

    n_images = len(images_file_lines[4:]) // 2

    data = {}
    image_names = []
    from scipy.spatial.transform import Rotation as R
    from skimage import transform
    for i in range(n_images):
        line_split = images_file_lines[4 + i * 2].split()
        image_id = int(line_split[0])

        q = np.array([float(x) for x in line_split[1: 5]]) # w, x, y, z
        t = np.array([float(x) for x in line_split[5: 8]])

        image_name = line_split[-1]
        image_names.append(image_name)

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R.from_quat(np.roll(q, -1)).as_matrix()
        extrinsic_matrix[:3, 3] = t
        # 
        # calib_matrix2 = np.identity(4)
        # calib_matrix2[:2,:2]=intrinsic_matrixs[i][:2,:2]
        # calib_matrix2[:2,3]=intrinsic_matrixs[i][:2,2]
        P = intrinsic_matrixs[i]@extrinsic_matrix@scale_mat
        # 下面会cv2后处理得到内外参，可有可无
        # P = P[:3, :4]  
        # intrinsics, pose = load_K_Rt_from_P(None, P)   
        # P = intrinsics@np.linalg.inv(pose)

        points_list=transform.matrix_transform(pc,P)
        points_list = points_list/points_list[:,2][:,None]
        points_list=points_list.astype('int')[:,:2]

        img = cv2.imread(os.path.join(path_to_scene,"image",'img_{:04d}.png'.format(i)))
        for point in points_list:
            cv2.circle(img, point, 2, (0,255,0), 1)
        cv2.imwrite("tmp.png",img)    
        
    
    #         Load camera
    cameras = camera_dict['arr_0']
    for i in range(len(cameras)):
        world_mat = cameras[i]  
        P = world_mat @ scale_mat
        # P = P[:3, :4]  
        # intrinsics, pose = load_K_Rt_from_P(None, P)   
        # P = intrinsics@np.linalg.inv(pose)
        # P = intrinsics@pose
        points_list=transform.matrix_transform(pc,P)
        points_list = points_list/points_list[:,2][:,None]
        points_list=points_list.astype('int')[:,:2]
        img = cv2.imread(os.path.join(path_to_scene,"image",'img_{:04d}.png'.format(i)))
        for point in points_list:
            cv2.circle(img, point, 2, (0,255,0), 1)
        cv2.imwrite("tmp.png",img)     

def readjson(file):
    import json
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict

def test_blender_ortho_intrinsic(path_to_scene):
    """测试blender renderpeople脚本得到的正视投影内外参是否正确
    """ 
    # idx = readjson("/home/algo/yangxinhang/NeuralHaircut/preprocess_custom_data/front.json")["back"]
    pc = np.array(trimesh.load(os.path.join(path_to_scene, 'head_prior_wo_eyes.obj'),process=False).vertices)
    # pc = np.array(trimesh.load(os.path.join(path_to_scene, 'point_cloud_cropped.ply'),process=False).vertices)
    # pc=pc[idx,:]
    from glob import glob
    files = glob(os.path.join(path_to_scene,"camera","*"))
    # files = sorted(files,key=lambda x: int(os.path.basename(x).split('k')[-1].split('.')[0]))
    files = sorted(files,key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    from skimage import transform
    scale_mat = np.eye(4, dtype=np.float32)
    with open(os.path.join(path_to_scene,"scale.pickle"), 'rb') as f:
        trans = pickle.load(f)
        print('upload transform', trans, os.path.join(path_to_scene,"scale.pickle"))
        scale_mat[:3, :3] *= trans['scale']
        scale_mat[:3, 3] = np.array(trans['translation'])
    for file in files[28:]:
        camera_dict = np.load(file,allow_pickle=True)
        transform_matrix = camera_dict.item().get('transform')
        calib_matrix1 = camera_dict.item().get('calib')
        calib_matrix1[0,1]=0
        calib_matrix1[1,0]=0
        # transform_matrix = camera_dict.item().get('calib')
        # calib_matrix1 = camera_dict.item().get('transform')
        calib_matrix2 = np.identity(4)
        calib_matrix2[:2,:2]=calib_matrix1[:2,:2]
        calib_matrix2[:2,3]=calib_matrix1[:2,2]

        calib_matrix3 = np.identity(4)
        calib_matrix3[:2,:3]=calib_matrix1
        P = calib_matrix2@transform_matrix@scale_mat
        P = P[:3, :4]  
        intrinsics, pose = load_K_Rt_from_P(None, P)   
        P = intrinsics@np.linalg.inv(pose)
        
        points_list=transform.matrix_transform(pc,P)
        
        # points_list = points_list/points_list[:,2][:,None]
        # points_list=transform.matrix_transform(points_list,calib_matrix3)
        points_list=points_list.astype('int')[:,:2]
        # points_list[:,:1]+=512
        # points_list[:,2]=512-points_list[:,2]
        img = np.zeros((1024, 1024, 3), np.uint8)
        img = cv2.imread(os.path.join(path_to_scene,"image",os.path.basename(file).split('.')[0]+'.png'))
        for point in points_list:
            cv2.circle(img, point, 2, (0,255,0), 1)
        cv2.imwrite("tmp.png",img)   
           
def test_blender_pers_intrinsic(path_to_scene):
    """测试blender renderpeople脚本得到的正视投影内外参是否正确
    """    
    
    # idx = readjson("/home/algo/yangxinhang/NeuralHaircut/preprocess_custom_data/front.json")["back"]
    pc = np.array(trimesh.load(os.path.join(path_to_scene, 'people_normalize.obj'),process=False).vertices)
    # pc = np.array(trimesh.load(os.path.join(path_to_scene, 'point_cloud_cropped.ply'),process=False).vertices)
    # pc=pc[idx,:]
    from glob import glob
    files = glob(os.path.join(path_to_scene,"camera","*"))
    # files = sorted(files,key=lambda x: int(os.path.basename(x).split('k')[-1].split('.')[0]))
    files = sorted(files,key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    from skimage import transform
    scale_mat = np.eye(4, dtype=np.float32)
    # with open(os.path.join(path_to_scene,"scale.pickle"), 'rb') as f:
    #     trans = pickle.load(f)
    #     print('upload transform', trans, os.path.join(path_to_scene,"scale.pickle"))
    #     scale_mat[:3, :3] *= trans['scale']
    #     scale_mat[:3, 3] = np.array(trans['translation'])
    for i,file in enumerate(files[:]):
        camera_dict = np.load(file,allow_pickle=True)
        transform_matrix = camera_dict.item().get('transform')
        calib_matrix1 = camera_dict.item().get('calib')
        calib_matrix1[0,1]=0
        calib_matrix1[1,0]=0
        # transform_matrix = camera_dict.item().get('calib')
        # calib_matrix1 = camera_dict.item().get('transform')
        calib_matrix2 = np.identity(4)
        #ortho
        calib_matrix2[:2,:2]=calib_matrix1[:2,:2]
        calib_matrix2[:2,3]=calib_matrix1[:2,2]
        #pers
        calib_matrix3 = np.identity(4)
        calib_matrix3[:2,:3]=calib_matrix1[:2,:3]
    
        P = calib_matrix3@transform_matrix@scale_mat
        points_list=transform.matrix_transform(pc,P)
        points_list = points_list/points_list[:,2][:,None]
        # 先用投影矩阵，再转到图像
        # project = np.array([[2.8000, 0.0000,  0.0000,  0.0000],
        #                     [0.0000, 2.8000,  0.0000,  0.0000],
        #                     [0.0000, 0.0000, -1.0002, -0.2000],
        #                     [0.0000, 0.0000, -1.0000,  0.0000]])
        # c = np.array([[-512, 0.00000000e+00, 0,5.12000000e+02],
        # [0.00000000e+00, -512, 0,5.12000000e+02],
        # [0.00000000e+00, 0.00000000e+00, 1.00000000e+00,0],[0.00000000e+00, 0.00000000e+00, 0,1.00000000e+00]])
        # calib = c@project
        # P = c@project@transform_matrix@scale_mat
        # points_list=transform.matrix_transform(pc,P)
        # 以下调用cv的分解，可有可无
        # P = P[:3, :4]  
        # intrinsics, pose = load_K_Rt_from_P(None, P)   
        # P = intrinsics@np.linalg.inv(pose)
        # points_list=transform.matrix_transform(pc,P)
        
        points_list=points_list.astype('int')[:,:2]
        # points_list[:,:1]+=512
        # points_list[:,2]=512-points_list[:,2]
        img = np.zeros((1024, 1024, 3), np.uint8)
        img = cv2.imread(os.path.join(path_to_scene,"image",os.path.basename(file).split('.')[0]+'.png'))
        for point in points_list:
            cv2.circle(img, point, 2, (0,255,0), 1)
        cv2.imwrite(f"tmp_{i}.png",img)      
if __name__=="__main__":
    path_to_scene = "/home/algo/yangxinhang/NeuralHaircut/implicit-hair-data/data/monocular/person_0_1"
    # test_colmap_intrinsic(path_to_scene)
    path_to_scene = "/home/algo/yangxinhang/NeuralHaircut/implicit-hair-data/data/monocular/render_1_pers"
    test_blender_pers_intrinsic(path_to_scene)
    path_to_scene = "/home/algo/yangxinhang/NeuralHaircut/implicit-hair-data/data/monocular/render_1"
    # test_blender_ortho_intrinsic(path_to_scene)