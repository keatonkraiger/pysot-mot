import os
import re
import imageio
from glob import glob
from PIL import Image

SAVE_FORMAT = 'gif'
video_name = 'ants1'



image_folder = os.path.join(os.getcwd(), 'demo/demo_images/')


#images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#images.sort(key=lambda var:[int(x) if x.isdigit() else x for
#        x in re.findall(r'[^0-9]|[0-9]+', var)])

if SAVE_FORMAT == 'gif':
    images = []

    images = glob("demo/demo_images/*.jpg")
    images.sort(key=lambda var:[int(x) if x.isdigit() else x for
        x in re.findall(r'[^0-9]|[0-9]+', var)])

    img, *imgs = [Image.open(f) for f in images]
    img.save(fp='demo/output/{}.gif'.format(video_name), format='GIF', append_images=imgs,
            save_all=True, duration=10, loop=0)



