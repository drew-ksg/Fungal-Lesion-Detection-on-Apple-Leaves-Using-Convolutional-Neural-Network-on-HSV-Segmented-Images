"""HSV segmentation: picture -> HSV -> thresholds -> binary mask -> masked picture.

Typical use:

    import cv2
    from src.hsv import segment

    bgr = cv2.imread("leaf.jpg")
    result = segment.segment_leaf(bgr)
    result.masked      # original picture with only suspected lesion pixels kept
    result.boxes       # candidate lesion boxes (x, y, w, h) for debugging

Note: OpenCV loads images in BGR order, not RGB. Every function here expects
BGR, which is what cv2.imread returns.
"""


import cv2
#import numpy as np
from .segment import segment_leaf


#image = cv2.imread(r"C:\Users\arivera106\Senior Project\apple-lesions\data\raw\plantvillage_apple\train\Apple___Apple_scab\0672ab32-9fce-41f3-ae69-e39c48a0a292___FREC_Scab_3347.JPG")
image_paths = [
    r"C:\Users\arivera106\Senior Project\apple-lesions\data\raw\plantvillage_apple\train\Apple___Apple_scab\0672ab32-9fce-41f3-ae69-e39c48a0a292___FREC_Scab_3347.JPG",
    r"C:\Users\arivera106\Senior Project\apple-lesions\data\raw\plantvillage_apple\train\Apple___Black_rot\02168189-aa75-4284-a7f0-8ca5901ea783___JR_FrgE.S_2948.JPG",
    r"C:\Users\arivera106\Senior Project\apple-lesions\data\raw\plantvillage_apple\train\Apple___Cedar_apple_rust\112cfc79-00ab-4920-a235-977c134bdb5d___FREC_C.Rust_4365.JPG",
    r"C:\Users\arivera106\Senior Project\apple-lesions\data\raw\plantvillage_apple\train\Apple___healthy\02847045-4405-467c-9ccb-7ba2ab1d574c___RS_HL_5943.JPG",
]

for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    
    result = segment_leaf(image)

    cv2.imshow("Original Image", image)
    cv2.imshow("Binary mask", result.lesion_mask)
    cv2.imshow("Masked result", result.masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







#hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


#cv2.imshow("Original Image", image)
#cv2.imshow("HSV Image", hsv_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#lower_dark = np.array([0, 0, 0])
#upper_dark = np.array([179, 255, 50])

#mask = cv2.inRange(hsv_image, lower_dark, upper_dark)
#result = cv2.bitwise_and(image, image, mask=mask)
#cv2.imshow("Mask", mask)
#cv2.imshow("Result", result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


