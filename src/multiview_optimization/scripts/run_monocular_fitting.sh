python fit.py --conf confs/train_person_7.conf --batch_size 1 --train_rotation True --save_path ./experiments/fit_person_7_bs_1

python fit.py --conf confs/train_person_7.conf --batch_size 5 --train_rotation True --save_path  ./experiments/fit_person_7_bs_5 --checkpoint_path ./experiments/fit_person_7_bs_1/opt_params

python fit.py --conf confs/train_person_7_.conf --batch_size 20 --train_rotation True --train_shape True --save_path  ./experiments/fit_person_7_bs_20_train_rot_shape  --checkpoint_path ./experiments/fit_person_7_bs_5/opt_params