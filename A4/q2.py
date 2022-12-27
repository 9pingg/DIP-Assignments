import scipy as s
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import cv2

def gauss(shape,mu=2):
    r,c = [(sh-1)/2 for sh in shape]
    x,y = np.ogrid[-r:r+1,-c:c+1]
    calc = np.exp( -(x*x + y*y) / (2*mu*mu) )
    sum_calc = calc.sum()
    calc = calc / sum_calc
    return calc

def zero_crossing(img, threshold = 0.04):
    res = np.zeros(img.shape)
    xx = res.shape[0]
    yy = res.shape[1]

    for c in range(1, xx - 1):
        for r in range(1, yy - 1):
            p = img[c, r]    
            n = img[c-1:c+2, r-1:r+2]
            
            max_val = n.max()
            min_val = n.min()

            if(p > 0):
                if(min_val < 0): todo_update = True
                else: todo_update = False
            else:
                if(max_val > 0): todo_update = True
                else: todo_update = False
            diff = max_val - min_val
            if (diff > threshold) and todo_update:
                res[c, r] = 1
    return res
clean_img = cv2.imread("x5.bmp", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("noise_img_10.bmp", cv2.IMREAD_GRAYSCALE)
cv2.imshow("clean img",clean_img.astype('uint8'))
cv2.imshow("noise img",img.astype('uint8'))

# q2 a)

LoG_img = cv2.Laplacian(cv2.GaussianBlur(img, (3, 3), 2), -1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
LoG_img = np.clip(LoG_img, 0, 255)
cv2.imshow("Log response img",LoG_img.astype('uint8'))

threshold = np.absolute(LoG_img).mean() * 0.04
res = zero_crossing(LoG_img, threshold)
cv2.imshow("zero crossing LoG",res)






# q2 b)

# getting 1-D Gaussian filters 
g_x = gauss((1,3))
g_y = gauss((3,1))
print("gaussian filter ", g_x, g_y)
# print(s.signal.get_window(('gaussian',2),3))

img_X = s.signal.convolve2d(img,g_x)
img_XY = s.signal.convolve2d(img_X, g_y)
img_XY = np.clip(img_XY, 0, 255)

# cv2.imshow("noisy image", img.astype('uint8'))
cv2.imshow("gaussian response image", img_XY.astype('uint8'))

l = np.array([
    [-1, -1, -1], 
    [-1,  8, -1], 
    [-1, -1, -1]
])


# getting 1-D Laplacian filters 

lx = np.array([[1,-2,1]])
ly = np.array([[1],[-2],[1]])

# l1 = np.array([
#     [0,	0,	-1,	0,	0],
# [0,	-1,	-2,	-1,	0],
# [-1,-2,	16,	-2,	-1],
# [0,	-1,	-2,	-1,	0],
# [0,	0,	-1,	0,	0]
# ])

img_XY_lx = s.signal.convolve2d(img_XY,lx)
img_XY_lx_ly = s.signal.convolve2d(img_XY_lx, ly)
img_XY_lx_ly = np.clip(img_XY_lx_ly, 0, 255)

img_XYL = s.signal.convolve2d(img_XY, l)
img_XYL = np.clip(img_XYL, 0, 255)
cv2.imshow("laplacian response image", img_XYL.astype('uint8'))


threshold = np.absolute(img_XYL).mean() * 0.04
res = zero_crossing(img_XYL, threshold)
img_XYL = cv2.resize(img_XYL, (512,512))
cv2.imshow("zero crossing q2 b)",res)
t = abs(LoG_img-img_XYL)
t = (t - np.amin(t))*255 / np.amax(t) 
print("abs sum of differences: ", sum(sum(t)), t)

cv2.waitKey(0)
cv2.destroyAllWindows()







