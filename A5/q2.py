import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


orignal_img = cv2.imread("lena.jpg")
img = cv2.cvtColor(orignal_img,cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(orignal_img,cv2.COLOR_BGR2LAB)

l,a,b = cv2.split(lab)
cv2.imshow("l", l)
cv2.imshow("a", a)
cv2.imshow("b", b)

img = np.copy(l)
normalize = np.copy(l)
max_val = np.amax(img)
print(max_val)
normalize = img / max_val
threshold_values = [int(0.25 * max_val), int(0.5 * max_val)]

def make_hist(img):
   m,n = img.shape
   hist = np.zeros(256)
   for i in range(m):
      for j in range(n):
         hist[img[i,j]] += 1
   return hist


def weight_probablity(start, end):
    p = 0
    for i in range(start, end):
        p = p + count_pixel[i]
    return p


def mean(f, l):
    mean = 0
    start = f
    end = l
    p = weight_probablity(start, end)
    for j in range(f, l):
        mean += count_pixel[j] * (j / max_val)
    weighted_mean = mean/p # float
    return weighted_mean


def variance(first, last):
    p = weight_probablity(first, last)
    weighted_variance = 0
    weighted_mean = mean(first, last)
    for i in range(first, last):
        temp = ((i/max_val) - weighted_mean) **2
        weighted_variance += temp * count_pixel[i]
    weighted_variance = weighted_variance / p
    return weighted_variance
            

def threshold(count_pixel):
    #cnt = countPixel(h)
    #print(cnt)
    count = float(img.size)
    print(img.size)
    min_t = 255
    min_withinvar = 1
    for i in threshold_values:
        weight_back = weight_probablity(0, i) / count
        variance_back = variance(0, i)
        mean_back = mean(0, i)
        
        weight_f = weight_probablity(i, len(count_pixel)) / count
        variance_f = variance(i, len(count_pixel))
        mean_f = mean(i, len(count_pixel))
        
        within_class_variance = weight_back * (variance_back) + weight_f * (variance_f)
        
        print('T= ', round(i/max_val, 2))
        print("weight_back= ", weight_back)
        print("mean_back= ", mean_back)
        print("variance_back= ",variance_back)
       
        print("weight_f= ", weight_f)
        print("mean_f= ", mean_f)
        print("variance_f= ", variance_f)
        print('within class variance= ', within_class_variance)
        
        if(within_class_variance < min_withinvar):
            min_withinvar = within_class_variance
            min_t = i
    print("threshold for which it is minimum: ", min_t/max_val)
    return min_t/max_val

def thresh(img, threshold):
    threshold = threshold * max_val
    row, col = img.shape 
    for i in range(0,row):
      for j in range(0,col):
        if(img[i,j] > threshold): img[i,j] = 255
        else: img[i,j] = 0
    cv2.imshow("res", img.astype('uint8'))
  
count_pixel = make_hist(img)
t = threshold(count_pixel)
thresh(img, t)
cv2.waitKey(0)

