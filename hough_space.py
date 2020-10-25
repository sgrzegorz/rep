import traceback

import cv2, sys, os
import numpy as np
from matplotlib import pyplot as plt



# Read image
img = cv2.imread('circles.png', cv2.IMREAD_GRAYSCALE)


import canny_edge_detector as ced
detector = ced.cannyEdgeDetector(img, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)

img = detector.detect()

# cv2.imshow("canny",img)
# cv2.waitKey(0)

param1=80;param2=44;minRadius=30;maxRadius=50
radius = np.arange(minRadius,maxRadius,1)
# fi_range = [0,45,90,135,180,225,270,315]
fi_range = range(0,360,10)

H = np.zeros(img.shape + (maxRadius,)) #3D matrix
for pixel in np.ndindex(img.shape[:2]):
    for r in radius:
        # for fi in range(0,360):
        for fi in fi_range:
            x= pixel[0]
            y = pixel[1]
            a = x - r*np.cos(np.deg2rad(fi))
            b = y + r*np.cos(np.deg2rad(fi))
            try:
                a =int(a);b=int(b);r=int(r)
                if  (0<=a<H.shape[0] and 0<= b < H.shape[1]):
                    H[a,b,r] +=1
            except Exception  as e:
                traceback.print_exc()


    if pixel[0]%10==0 and pixel[1]==0:
        print(f' {pixel[0]}/{img.shape[0]}')
print('Hough finished')

H = np.amax(H,axis=2)
plt.imshow(H, cmap='hot', interpolation='nearest')
plt.show()

cv2.imshow("result",img)
cv2.waitKey(0)
