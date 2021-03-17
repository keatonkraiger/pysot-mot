
import cv2
import torch
import numpy as np
from glob import glob
from PIL import Image

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--images_path', type=str,
        default='/Users/keaton/Desktop/pysot-master/results', help='path to images')

args = parser.parse_args()


img, *imgs = [Image.open(f) for f in sorted(glob("{}/*.jpg".format(args.images_path)))]
img.save(fp='result.gif', format='GIF', append_images=imgs,
                save_all=True, duration=200, loop=0)


