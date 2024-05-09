import trimesh
import numpy as np
import pickle

import os



def main(args):
    
    path_to_scene = os.path.join(args.path_to_data, args.scene_type, args.case)

    pc = np.array(trimesh.load(os.path.join(path_to_scene, 'point_cloud_cropped.ply')).vertices)

    translation = (pc.min(0) + pc.max(0)) / 2
    scale = np.linalg.norm(pc - translation, axis=-1).max().item() / 1.1

    tr = (pc - translation) / scale
    assert tr.min() >= -1 and tr.max() <= 1#tr.min() 0.92-0.93之间

    print('Scaling into the sphere', tr.min(), tr.max())

    d = {'scale': scale,
        'translation': list(translation)}

    ply=trimesh.Trimesh(vertices=tr)
    ply.export(os.path.join(path_to_scene, 'point_cloud_cropped_normalize.ply'))
    with open(os.path.join(path_to_scene, 'scale.pickle'), 'wb') as f:
        pickle.dump(d, f)


def scale_render_people(path_to_scene):
    
    # path_to_scene = os.path.join(args.path_to_data, args.scene_type, args.case)
    
    model=trimesh.load(os.path.join(path_to_scene, 'head_prior_wo_eyes_render_1_pers1.obj'),process=False)
    # model=trimesh.load(os.path.join(path_to_scene, 'head_prior_wo_eyes_render_2.obj'),process=False)

    pc = np.array(model.vertices)
    translation = (pc.min(0) + pc.max(0)) / 2
    scale = np.linalg.norm(pc - translation, axis=-1).max().item() / 0.5
    tr = (pc - translation) / scale
    assert (tr.min() >= -1 and tr.max() <= 1)#tr.min() 0.92-0.93之间
    print('Scaling into the sphere', tr.min(), tr.max())
    d = {'scale': scale,
        'translation': list(translation)}
   
    model.vertices=tr
    model.export(os.path.join(path_to_scene, 'head_prior_wo_eyes.obj'))
    with open(os.path.join(path_to_scene, 'scale.pickle'), 'wb') as f:
        pickle.dump(d, f)


if __name__ == "__main__": 
    # 渲染renderpeople的gt数据
    path_to_scene = "/home/algo/yangxinhang/NeuralHaircut/implicit-hair-data/data/monocular/render_1_pers1"
    scale_render_people(path_to_scene)
    import argparse
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--case', default='person_0', type=str)
    parser.add_argument('--scene_type', default='monocular', type=str)
    
    parser.add_argument('--path_to_data', default='./implicit-hair-data/data/', type=str) 

    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)