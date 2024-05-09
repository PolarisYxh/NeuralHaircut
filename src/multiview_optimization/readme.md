Download pre-trained models and data from [SMPLX](https://smpl-x.is.tue.mpg.de/)  and [PIXIE](https://pixie.is.tue.mpg.de/) projects.

For more information please follow [PIXIE installation](https://github.com/yfeng95/PIXIE/blob/master/Doc/docs/getting_started.md).

For multiview optimization you need to have the following files ```SMPL-X__FLAME_vertex_ids.npy, smplx_extra_joints.yaml, SMPLX_NEUTRAL_2020.npz``` and change a path to them in ```./utils/config.py```


Note, that you need to obtain  [PIXIE initialization](https://github.com/yfeng95/PIXIE) for shape, pose parameters and save it as a dict in ```initialization_pixie``` file (see the structure in [example scene](../../example) for convenience). 

Furthermore, use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to obtain 3d keypoints or use only [FaceAlignment](https://github.com/1adrianb/face-alignment) loss in optimization process.


To obtain FLAME prior run:

```bash
cd src/multiview_optimization
python PIXIE/demos/demo_fit_face.py --input_path implicit-hair-data/data/monocular/person_7/image -save_path implicit-hair-data/data/monocular/person_7/pixie
#修改 run_monocular_fitting.sh和multiview_optimization/conf下的conf文件
bash scripts/run_monocular_fitting.sh

```
To visualize the training process:

```bash
tensorboard --logdir ./experiments/EXP_NAME
```

After training put obtained FLAME prior NeuralHaircut/src/multiview_optimization/experiments/fit_person_7_bs_20_train_rot_shape/mesh/mesh.obj into the dataset folder ```./implicit-hair-data/data/SCENE_TYPE/CASE/head_prior.obj```.

check in meshlab if ```./implicit-hair-data/data/SCENE_TYPE/CASE/head_prior.obj and point_cloud_cropped_normalize.ply``` almost same,if so, steps before is right!