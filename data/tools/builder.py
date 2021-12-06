import os
from data.tools.ROIExtracter.Pro_SLIC import SLIC
import numpy as np
import cv2
import math

current_path = os.path.abspath('.')


def skimage2opencv(src):
    src *= 255
    src.astype(int)
    cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    return src


def opencv2skimage(src):
    cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src.astype(np.float32)
    src /= 255
    return src



def build_processer(img, name):
    template = cv2.imread(os.path.join(current_path, '../config/template_straight.PNG'))
    hyp_config = os.path.join(current_path, f'../config/{name}.yaml')
    if name == "SLIC":
        # load config from yaml file
        if type(img) == 'ndarray':
            img = opencv2skimage(img)
        return SLIC(img, hyp_config)
    if name == "GnodeSim":
        pass
    if name == 'CL_temp':
        pass

# if __name__ == '__main__':
