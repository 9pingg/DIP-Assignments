import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math

def getHue(R,G,B,hue):
    m = R.shape[0]
    n = R.shape[0]
    # hue = 0.5 * ((R - G) + (R - B)) / \
    #                     math.sqrt((R - G)**2 + ((R - B) * (G - B)))
    # hue = math.acos(hue)
    for i in range(0, m):
        for j in range(0, n):
            hue[i][j] = 0.5 * ((R[i][j] - B[i][j]) + (R[i][j] - G[i][j])) / \
                        math.sqrt((R[i][j] - G[i][j])**2 +((G[i][j] - B[i][j])*(R[i][j] - B[i][j])))
            hue[i][j] = math.acos(hue[i][j])

    for i in range(0, m):
        for j in range(0, n):
            if B[i][j] <= G[i][j]:
                hue[i][j] = hue[i][j]
            else:
                hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]


    return hue

def getSaturation(R, G, B):
    min_val = np.minimum(R, G)
    min_val = np.minimum(B, min_val)
    s = 1 - (3 / (R + G + B + 0.001) * min_val)
    return s
def getIntensity(R, G, B):
    intensity = np.divide(R+G+B, 3)
    return intensity


img = cv2.imread("q1.tiff")
cv2.imshow("img", img)

BGR = img/255

R = BGR[:,:,2]
G = BGR[:,:,1]
B = BGR[:,:,0]

# hue
hue = np.copy(R)
hue = getHue(R, G, B, hue)
saturation = getSaturation(R, G, B)
intensity = getIntensity(R, G, B)
hsi = cv2.merge((hue, saturation, intensity))
cv2.imshow("hsi", hsi)
cv2.imshow("hue", hue)
cv2.imshow("saturation", saturation)
cv2.imshow("intensity", intensity)

cv2.waitKey(0)
