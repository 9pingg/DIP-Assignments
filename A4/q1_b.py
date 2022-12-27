import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy

def addAGWN(img, mean, sigma):
	noise = np.copy(img)
	noise = np.random.normal(mean, sigma, img.shape) 
	return noise

def filter(shape, K, lamb, G, L, H, label):
    m, n = shape
    H_con = H.conjugate()
    H_mag_sqr = np.abs(H)**2
    L_mag_sqr = np.abs(L)**2
    
    Filter = H_con / ((H_mag_sqr + K) * (lamb*(L_mag_sqr) + 1))

    

    F_cap = Filter*G
    f_cap = np.fft.ifft2(F_cap)
    f_cap = np.absolute(f_cap)
    f_cap = np.clip(f_cap, 0, 255)


    psnr = 10*np.log10(255*255/np.mean((img-f_cap)**2))
    print("PSNR: ", psnr)
    cv2.imshow(label, f_cap.astype('uint8'))
    
def gauss(shape,mu=2):
    r,c = [(sh-1)/2 for sh in shape]
    x,y = np.ogrid[-r:r+1,-c:c+1]
    calc = np.exp( -(x*x + y*y) / (2*mu*mu) )
    sum_calc = calc.sum()
    calc = calc / sum_calc
    return calc


img = cv2.imread("x5.bmp", cv2.IMREAD_GRAYSCALE)
noise = addAGWN(img, 0, 25)
noise_img = img + noise
noise_img = np.clip(noise_img, 0, 255)
m,n = img.shape[0], img.shape[1]
cv2.imshow("input image", img)
cv2.imshow("noise image", noise_img.astype('uint8'))
cv2.imwrite("noise_img.bmp", noise_img.astype('uint8'))


#g = np.pad(noise_img, ((0, m), (0, n)))
G = np.fft.fft2(noise_img)
L = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
l = np.pad(L, ((0, m-3), (0, n-3)))
L = np.fft.fft2(l)
h = scipy.signal.gaussian(3, 1).reshape(3, 1)
h = np.dot(h, h.transpose())
h /= np.sum(h)
kernel = h
kernel = np.pad(kernel, [(0, m - kernel.shape[0]), (0, n - kernel.shape[1])], 'constant')
H = np.fft.fft2(kernel)



K,l = 0.07,2

filter((m, n), K, l, G, L, H,"Denoised Image custom filter")
filter((m, n), 0.75, 0, G, L, H,"Denoised Image wiener filter")

cv2.waitKey(0)
cv2.destroyAllWindows()

