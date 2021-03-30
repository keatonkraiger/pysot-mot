from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse

import cv2
import torch
import re
import numpy as np
from glob import glob
from PIL import Image
import imageio

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.sampling.multi_template import multi_template

torch.set_num_threads(1)

SAVE_FORMAT = "mp4"

parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument("--config", type=str, help="config file")
parser.add_argument("--snapshot", type=str, help="model name")
parser.add_argument("--video_name", default="", type=str, help="videos or image files")
parser.add_argument(
    "--save", action="store_true", help="whether to save resulting video"
)

args = parser.parse_args()


def generate_video(video_name):
    image_folder = os.path.join(os.getcwd(), "demo/demo_images/")

    images = glob("demo/demo_images/*.jpg")
    images.sort(
        key=lambda var: [
            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
        ]
    )

    if SAVE_FORMAT == "gif":
        img, *imgs = [Image.open(f) for f in images]
        img.save(
            fp="demo/output/{}.gif".format(video_name),
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=10,
            loop=0,
        )

    else:
        # frame = cv2.imread(os.path.join(image_folder, images[0]))
        print(images[0])
        frame = cv2.imread(images[0])

        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video = cv2.VideoWriter(
            "demo/output/{}.mp4".format(video_name), fourcc, 5, (width, height)
        )

        # images.sort(key=lambda var:[int(x) if x.isdigit() else x for
        #    x in re.findall(r'[^0-9]|[0-9]+', var)])

        for image in images:
            # video.write(cv2.imread(os.path.join(image_folder, image)))
            video.write(cv2.imread(image))

        cv2.destroyAllWindows()
        video.release()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith("avi") or video_name.endswith("mp4"):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, "*.jp*"))
        images = sorted(images, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device("cuda" if cfg.CUDA else "cpu")

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(
        torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu())
    )
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    i = 0
    if args.video_name:
        video_name = args.video_name.split("/")[-1].split(".")[0]
    else:
        video_name = "webcam"
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if "polygon" in outputs:
                polygon = np.array(outputs["polygon"]).astype(np.int32)
                cv2.polylines(
                    frame, [polygon.reshape((-1, 1, 2))], True, (0, 255, 0), 3
                )
                mask = (outputs["mask"] > cfg.TRACK.MASK_THERSHOLD) * 255
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs["bbox"]))
                """
                rec_free = frame.copy()
                cropped_image = rec_free[
                    bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]
                ]
                # cv2.imshow("cropped", cropped_image)
                """
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (0, 255, 0),
                    3,
                )
            cv2.imshow(video_name, frame)
            """
            multi_template(rec_free, cropped_image, 0.8)
            cv2.imshow("cropped", cropped_image)
            cv2.waitKey(0)
            """
            if args.save:
                if SAVE_FORMAT != "gif":
                    cv2.imwrite("demo/demo_images/{}.jpg".format(i), frame)
                    i += 1
                elif SAVE_FORMAT == "gif" and i < 50:
                    cv2.imwrite("demo/demo_images/{}.jpg".format(i), frame)
                    i += 1

            # if(video_name == 'webcam' and i == 50):
            #    break
            cv2.waitKey(40)

    if args.save:
        generate_video(video_name)


if __name__ == "__main__":
    main()
