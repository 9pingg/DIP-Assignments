import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def set_values(g_dir):
    m = g_dir.shape[0]
    n = g_dir.shape[1]
    res = np.zeros((m, n))
    for i in range(g_dir.shape[0]):
        for j in range(g_dir.shape[1]):
            if 0 <= g_dir[i, j] <= 22.5:
                res[i, j] = 0
            elif 22.5 <= g_dir[i, j] <= 67.5:
                res[i, j] = 1
            elif 67.5 <= g_dir[i, j] <= 122.5:
                res[i, j] = 2
            elif 112.5 <= g_dir[i, j] <= 157.5:
                res[i, j] = 3
            elif 157.5 <= g_dir[i, j] <= 202.5:
                res[i, j] = 0
            elif 202.5 <= g_dir[i, j] <= 247.5:
                res[i, j] = 1
            elif 247.5 <= g_dir[i, j] <= 292.5:
                res[i, j] = 2
            elif 292.5 <= g_dir[i, j] <= 337.5:
                res[i, j] = 3
            elif 337.5 < g_dir[i, j] < 360:
                res[i, j] = 0
    return res
    
def non_max_suppression(set_value, g_mag, g_dir):
    res = np.zeros(set_value.shape)
    m, n = np.shape(set_value)
    m = m - 1
    n = n - 1
    for i in range(m):
        for j in range(n):
            if set_value[i,j] == 0:
                if  g_mag[i,j+1] < g_mag[i,j] or g_mag[i,j-1]< g_mag[i,j]:
                    res[i,j] = g_dir[i,j]
                else:
                    res[i,j] = 0
            if set_value[i,j]==1:
                if  g_mag[i,j] >= g_mag[i+1,j-1] or g_mag[i,j] >= g_mag[i-1,j+1]:
                    res[i,j] = g_dir[i,j]
                else:
                    res[i,j] = 0       
            if set_value[i,j] == 2:
                if   g_mag[i+1,j] <= g_mag[i,j] or g_mag[i,j] >= g_mag[i-1,j]:
                    res[i,j] = g_dir[i,j]
                else:
                    res[i,j] = 0
            if set_value[i,j] == 3:
                if g_mag[i,j] >= g_mag[i-1,j-1] or g_mag[i+1,j+1] <= g_mag[i,j]:
                    res[i,j] = g_dir[i,j]
                else:
                    res[i,j] = 0
    return res

orignal_img = cv2.imread("q1.tiff")
img = cv2.cvtColor(orignal_img,cv2.COLOR_BGR2GRAY)
img=cv2.GaussianBlur(img,(3,3),0.2)
gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0,3)
gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1,3)
cv2.imshow("gx", np.clip(gx, 0, 255).astype('uint8'))
cv2.imshow("gy", np.clip(gy, 0, 255).astype('uint8'))

Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
fx = cv2.filter2D(img, -1, Gx)
fy = cv2.filter2D(img, -1, Gy)

g_mag = np.sqrt(gx * gx + gy * gy)
g_mag *= 100.0 / np.amax(g_mag) # 100
cv2.imshow("g_mag", np.clip(g_mag, 0, 255).astype('uint8'))
g_dir = np.zeros((gx.shape[0], gx.shape[1]))
g_dir = np.rad2deg(np.arctan2(gy, gx)) + 180
cv2.imshow("g_dir", np.clip(g_dir, 0, 255).astype('uint8'))

quantized = set_values(g_dir)
nms = non_max_suppression(quantized, g_dir, g_mag)
cv2.imshow("nms", np.clip(nms, 0, 255).astype('uint8'))
cv2.waitKey(0)
