import cv2
from pysot.utils.non_max_suppression import non_max_suppression

import numpy as np


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
    # cv2.imshow("After NMS", image)
    # cv2.waitKey(0)
