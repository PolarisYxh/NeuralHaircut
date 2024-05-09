import cv2
import os
import numpy as np
if __name__=="__main__":
    dir_name = "/home/algo/yangxinhang/NeuralHaircut/implicit-hair-data/data/monocular/person_5/image"
    dir_name1 = "/home/algo/yangxinhang/NeuralHaircut/implicit-hair-data/data/monocular/person_5/mask"
    
    save_dir = "/home/algo/yangxinhang/NeuralHaircut/implicit-hair-data/data/monocular/person_5/human"
    file_names = os.listdir(dir_name)
    for file_name in file_names:
        img = cv2.imread(os.path.join(dir_name,file_name))
        mask = cv2.imread(os.path.join(dir_name1,file_name))
        mask=np.sum(mask,axis=2)
        img[mask==0] =[0,0,0]
        cv2.imwrite(os.path.join(save_dir,file_name),img)