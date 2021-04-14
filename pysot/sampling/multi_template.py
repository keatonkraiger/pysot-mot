import cv2
from scipy import misc, ndimage
from pysot.utils.non_max_suppression import non_max_suppression

import numpy as np


def multi_template(image, template, threshold=0.7):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    rotations = make_rotations(template_gray)

    total_rects = []
    clone = image.copy()

    for rot in rotations:
        (tH, tW) = rot.shape[:2]
        result = cv2.matchTemplate(image_gray, rot, cv2.TM_CCOEFF_NORMED)

        (yCoords, xCoords) = np.where(result >= threshold)
        rects = []

        # loop over the starting (x, y)-coordinates again
        for (x, y) in zip(xCoords, yCoords):
            # update our list of rectangles
            rects.append((x, y, x + tW, y + tH))

        total_rects.append(rects)
        # apply non-maxima suppression to the rectangles
        # pick = non_max_suppression(np.array(rects))
        # print("[INFO] {} matched locations *after* NMS".format(len(pick)))

        # loop over the final bounding boxes

    picks = []
    for i in range(len(total_rects)):
        vals = non_max_suppression(np.array(total_rects[i]))

        if len(vals):
            picks.append(non_max_suppression(np.array(total_rects[i])))

    for i in range(len(picks)):
        print(picks[i])
        startX, startY, endX, endY = picks[i][0]
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

    # for (startX, startY, endX, endY) in picks:
    #    # draw the bounding box on the image
    #    cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

    # show the output image
    cv2.imshow("After NMS", image)


"""
def multi_template(image, template, threshold=0.8):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    (tH, tW) = template.shape[:2]
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    (yCoords, xCoords) = np.where(result >= threshold)
    clone = image.copy()

    rects = []

    # loop over the starting (x, y)-coordinates again

    for (x, y) in zip(xCoords, yCoords):
        # update our list of rectangles
        rects.append((x, y, x + tW, y + tH))

    # apply non-maxima suppression to the rectangles
    pick = non_max_suppression(np.array(rects))
    # print("[INFO] {} matched locations *after* NMS".format(len(pick)))

    # loop over the final bounding boxes
    for (startX, startY, endX, endY) in pick:
        # draw the bounding box on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

    # show the output image
    cv2.imshow("After NMS", image)
"""


def make_rotations(img):
    rots = []
    for deg in range(0, 360, 10):
        tmpl = ndimage.rotate(img, deg)
        rots.append(tmpl)

    return rots