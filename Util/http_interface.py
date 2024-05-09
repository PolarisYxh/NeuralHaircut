# -- coding: utf-8 --**
import requests
import json
import base64
import json
import os
import cv2
import numpy as np
from numpy.linalg import norm as l2norm
def cvmat2base64(img_np, houzhui='.jpg'):
    #opencv的Mat格式转为base64
    image = cv2.imencode(houzhui, img_np)[1]
    base64_data = str(base64.b64encode(image))
    return base64_data[2:-1]


def base642cvmat(base64_data):
    #base64转为opencv的Mat格式
    imgData = base64.b64decode(base64_data)
    nparr = np.frombuffer(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img_np
def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict

def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)
class HairStepInterface:
    def __init__(self,rfolder) -> None:
        config_file = os.path.join(rfolder,"config_test.json")
        load_dict = readjson(config_file)['HairStep']
        self.url = load_dict["url"]
        self.rfolder = rfolder
    def request_HairStep(self, reqCode, mode, imgB64, is_test=False):
        url = f"{self.url}/{mode}"
        print(url,reqCode)
        post_data = {"reqCode": reqCode,  "imgFile": imgB64}

        if is_test:
            post_data['is_test']=is_test
        res = requests.post(url=url, data=json.dumps(post_data))
        result = json.loads(res.content)
        if (result["error"] == 0):
            # parts = result["detected_part"]
            hair_step = result['step']
            os.makedirs(f"{self.rfolder}/cache",exist_ok=True)
            # parsing = base642cvmat(parsing)
            return np.array(hair_step)
        else:
            print(f"reqCode:{result['reqCode']}\nerror:{result['error']}\nerrorInfo:{result['errorInfo']}")
            return None,None,None
class segmentAllInterface:
    def __init__(self,rfolder) -> None:
        config_file = os.path.join(rfolder,"config_test.json")
        load_dict = readjson(config_file)['segmentAll']
        self.url = load_dict["url"]
        self.rfolder = rfolder
    def request_faceParsing(self, reqCode, mode, imgB64, point_coords, point_labels, is_test=False):
        url = f"{self.url}/{mode}"
        print(url,reqCode)
        post_data = {"reqCode": reqCode,  "imgFile": imgB64, "point_coords": point_coords.tolist(), "point_labels": point_labels.tolist()}

        if is_test:
            post_data['is_test']=is_test
        res = requests.post(url=url, data=json.dumps(post_data))
        result = json.loads(res.content)
        if (result["error"] == 0):
            # parts = result["detected_part"]
            masks = result['masks']
            os.makedirs(f"{self.rfolder}/cache",exist_ok=True)
            # parsing = base642cvmat(parsing)
            return np.array(masks)
        else:
            print(f"reqCode:{result['reqCode']}\nerror:{result['error']}\nerrorInfo:{result['errorInfo']}")
            return None,None,None
class faceParsingInterface:
    def __init__(self,rfolder) -> None:
        config_file = os.path.join(rfolder,"config_test.json")
        load_dict = readjson(config_file)['faceParsing']
        self.url = load_dict["url"]
        self.rfolder = rfolder
        
    def request_faceParsing(self, reqCode, mode, imgB64, is_test=False):
        url = f"{self.url}/{mode}"
        print(url,reqCode)
        post_data = {"reqCode": reqCode,  "imgFile": imgB64}

        if is_test:
            post_data['is_test']=is_test
        res = requests.post(url=url, data=json.dumps(post_data))
        result = json.loads(res.content)
        if (result["error"] == 0):
            parts = result["detected_part"]
            parsing = result['segment_img']
            os.makedirs(f"{self.rfolder}/cache",exist_ok=True)
            
            parsing = base642cvmat(parsing)
            return parts,parsing
        else:
            print(f"reqCode:{result['reqCode']}\nerror:{result['error']}\nerrorInfo:{result['errorInfo']}")
            return None,None,None
class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        #for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property 
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender==1 else 'F'
        
class insightFaceInterface:
    def __init__(self,rfolder) -> None:
        config_file = os.path.join(rfolder,"config_test.json")
        load_dict = readjson(config_file)['insightFace']
        self.url = load_dict["url"]
        self.rfolder = rfolder
        
    def request_insightFace(self,  img, reqCode="", is_test=False):
        url = f"{self.url}/img"
        print(url,reqCode)
        imgB64 = cvmat2base64(img)
        post_data = {"reqCode": reqCode, "imgFile": imgB64, "imgType": 'jpg'}
        if is_test:
            post_data['is_test']=is_test
        res = requests.post(url=url, data=json.dumps(post_data))
        result = json.loads(res.content)
        faces=[]
        if (result["error"] == 0):
            imgNames=["clipImg","imgFile","avatarImg"]
            os.makedirs(f"{self.rfolder}/cache",exist_ok=True)
            for face in result['result'].items():
                for name in face[1]:
                    if isinstance(face[1][name], list):
                        face[1][name] = np.array(face[1][name])
                f=Face(face[1])
                faces.append(f)
            return faces
        else:
            print(f"reqCode:{result['reqCode']}\nerror:{result['error']}\nerrorInfo:{result['errorInfo']}")
            return faces