"""HSV segmentation: picture -> HSV -> thresholds -> binary mask -> masked picture.

Typical use:

    import cv2
    from src.hsv import segment

    bgr = cv2.imread("leaf.jpg")
    result = segment.segment_leaf(bgr)
    result.masked      # original picture with only suspected lesion pixels kept
    result.boxes       # candidate lesion boxes (x, y, w, h) to send to the CNN

Note: OpenCV loads images in BGR order, not RGB. Every function here expects
BGR, which is what cv2.imread returns.
"""


import cv2
import numpy as np


image = cv2.imread(r"C:\Users\arivera106\Senior Project\apple-lesions\data\raw\plantvillage_apple\train\Apple___Apple_scab\0672ab32-9fce-41f3-ae69-e39c48a0a292___FREC_Scab_3347.JPG")
if image is None:
    raise FileNotFoundError("Image not found or unable to load.")
else:
    print("Image loaded successfully.")


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


cv2.imshow("Original Image", image)
#cv2.imshow("HSV Image", hsv_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


lower_dark = np.array([0, 0, 0])
upper_dark = np.array([179, 255, 70])

mask = cv2.inRange(hsv_image, lower_dark, upper_dark)
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask", mask)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


