import numpy as np
import cv2

def padImage(img, kernel_size):
	if(kernel_size % 2 == 0): pad_size = kernel_size/2
	else: pad_size = (kernel_size-1)/2
	pad_size = int(pad_size)
	return np.pad(img, pad_size)

def padKernel(orig_img, kernel):
	x = orig_img.shape[0] - kernel.shape[0]
	y = orig_img.shape[1] - kernel.shape[1]
	pad_t = ( ( (x+1)//2 , x//2), ( (y+1)//2, y//2) )
	kernel = np.pad(kernel, pad_t, 'constant')
	return kernel


img = cv2.imread("x5.bmp", 0)
#padding image
img = padImage(img,5)
print(img.shape)
cv2.imshow("orignal image", img)

#padding kernel
kernel = np.ones((5,5)) / 25
kernel = padKernel(img,kernel)
kernel = np.fft.ifftshift(kernel)
#print(kernel)
F = np.fft.fft2(img)
W = np.fft.fft2(kernel)
#print(F.dtype, W.dtype)
# blur image
FW = F*W
blur_img = np.real(np.fft.ifft2(FW))
#blur_img = np.uint8(blur_img)
blur_img = np.clip(blur_img, 0, 255).astype('uint8')
cv2.imshow("blurred image", blur_img)

#mask
mask = F - FW
#print(mask.dtype)
# unsharp masked image
g = np.real(np.fft.ifft2(F + mask))
#g = np.uint8(g)
cv2.imshow("unsharped masked image", np.clip(g, 0, 255).astype('uint8'))

g = np.real(np.fft.ifft2(F + 4 * mask))
#g = np.uint8(g)
cv2.imshow("high boost filtered image", np.clip(g, 0, 255).astype('uint8'))
cv2.waitKey(0)


# size of padded image = m+kern_m-1  n+kern_n-1
# n+4 516