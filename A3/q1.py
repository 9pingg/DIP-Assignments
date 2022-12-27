import cv2
import numpy as np

def conv_img_boxfilter(img, box_filter):
	"""
	performs convolution of an image with a box filter.
	"""
	size = box_filter[0]*box_filter[1]
	kernel = np.ones(box_filter,np.float32)/size
	#return cv2.blur(img,box_filter)
	return cv2.filter2D(img,-1,kernel)


def getMask(orig_img, blur_img):
	return orig_img.astype(np.int32) - blur_img.astype(np.int32)

def unsharpMasking(f, k, mask):
	return f + k * mask

orig_img = cv2.imread("x5.bmp", 0)
cv2.imshow("orignal image", orig_img)
#orig_img = orig_img.astype(np.int32) 

box_filter = (5,5)
blur_img = conv_img_boxfilter(orig_img,box_filter)
cv2.imshow("blurred image", blur_img)

mask = getMask(orig_img,blur_img)
print(mask.min(), mask.max())
g = unsharpMasking(orig_img, 1, mask)
cv2.imshow("unsharped masked image", np.clip(g, 0, 255).astype('uint8'))

g = unsharpMasking(orig_img, 4, mask)
cv2.imshow("high boost filtered image", np.clip(g, 0, 255).astype('uint8'))
cv2.waitKey(0)
