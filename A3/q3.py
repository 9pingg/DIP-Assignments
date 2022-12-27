import numpy as np
import cv2
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import q
def addingNoise(k, orig_img):
	m,n = orig_img.shape[0]-1, orig_img.shape[1]-1
	for i in range(512):
		for j in range(6):
			if(orig_img[100*j][i]+k > 255): orig_img[100*j][i] = 255
			else: orig_img[100*j][i] += k
	return orig_img

def drawImage(title, img):
	f, ax = plt.subplots(figsize=(4,4))
	ax.imshow(img,cmap = "gray")
	ax.set_title(title)

def drawSpectrum(title, F_real):
	f, ax = plt.subplots(figsize=(4,4))
	ax.imshow(np.log(1+F_real), cmap='gray')
	ax.set_title(title);
	return None	

def designingFilter(radius, F_real, imgg, F):
	x = M // 2 
	y = N // 2
	F_real[x - radius: x + radius, y - radius: y+ radius] = 0
	f, ax = plt.subplots(figsize=(4,4))
	ax.imshow(np.log(1 + F_real), cmap='gray',extent=(-y, y, -x, x))
	ax.set_title('Filter')
	filterr = F_real < np.percentile(F_real, 98)
	filterr = np.fft.ifftshift(filterr) 
	F_dim = F.copy()
	F_dim = F_dim * filterr.astype(int)

	f, ax= plt.subplots(figsize=(4,4))
	ax.imshow(np.log10(1 + np.abs(F_dim)), cmap='gray')
	ax.set_title('denoised spectrum')

	image_filtered = np.real(np.fft.ifft2(F_dim))
	image_filtered = q.printt(image_filtered, imgg)
	f, ax1 = plt.subplots(figsize=(4,4))
	ax1.imshow(image_filtered, cmap="gray")
	ax1.set_title('denoised image');
	return image_filtered


def forval(imgg, k):
	noise_img = np.copy(imgg)
	noise_img = addingNoise(k, noise_img)
	cv2.imwrite("noise_img.jpg", noise_img)

	M,N = noise_img.shape
	drawImage("Orignal Image", img)
	drawImage("Noisy Image", noise_img)

	F = np.fft.fftn(noise_img) 
	F_real = np.abs(F)

	drawSpectrum("Spectrum magnitude of Noisy Image", F_real)
	F_real = np.fft.fftshift(F_real)
	drawSpectrum("Spectrum magnitude (Centered)", F_real)

	image_filtered = designingFilter(75, F_real, imgg, F)
	plt.show()

img = cv2.imread("cameraman.jpg", 0)
imgg = cv2.resize(img, (512, 512),interpolation = cv2.INTER_NEAREST)
M,N = imgg.shape
F_orig = np.abs(np.fft.fftn(imgg))
drawSpectrum("Spectrum magnitude Orignal Image", F_orig)
forval(imgg, 20)
forval(imgg, 30)
forval(imgg, 50)



