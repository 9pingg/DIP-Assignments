def printt(img, imgg):
	m,n = imgg.shape[0]-1, imgg.shape[1]-1
	for i in range(512):
		for j in range(6):
			img[100*j][i] = imgg[100*j][i]
	return img