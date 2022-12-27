import cv2
import numpy as np
def soln(mat, interpolation_factor_x, interpolation_factor_y):
    eps=1e-5
    orig_size = mat.shape
    m,n = int(interpolation_factor_x*orig_size[0]), int(interpolation_factor_y*orig_size[1])
    res = np.zeros((m,n))
    mat = np.pad(mat, (1,1))
    for i in range(m):
        for j in range(n):
            
            x = i/interpolation_factor_x
            y = j/interpolation_factor_y
            x1, y1 = round(x), round(y)
            x3, y3 = round(x), round(y+1)
            x2, y2 = round(x+1), round(y)
            x4, y4 = round(x+1), round(y+1)
            V = np.array([ mat[x1,y1], mat[x2,y2], mat[x3,y3], mat[x4,y4] ])
            X = np.array([ 
                [x1, y1, x1*y1, 1], 
                [x2, y2, x2*y2, 1],
                [x3, y3, x3*y3, 1],
                [x4, y4, x4*y4, 1] 
                ])
            inv_X = np.linalg.inv( X + eps*np.eye(4))
            A = np.dot( inv_X, V )
            a,b,c,d = A
            # v(x,y) = a*x + b*y + c*x*y + d
            pixel_value = a*x + b*y + c*x*y + d
            if pixel_value<0:
                pixel_value = 0
            if pixel_value>255:
                pixel_value = 255
            res[i,j] = pixel_value

    return res
mat = cv2.imread("x5.bmp", cv2.IMREAD_GRAYSCALE)
res = soln(mat,0.2,0.2)
res = res.astype('uint8')
cv2.imshow("original", mat)
cv2.imshow("interpolated", res)
cv2.imwrite("res.png", res)
cv2.waitKey(0)













