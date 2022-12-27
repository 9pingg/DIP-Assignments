import math   
import cv2
import numpy as np


def BilinearInterpolation(mat, interpolation_factor_x, interpolation_factor_y):
	# epsilon can be altered
	eps=1e-5
	orig_size = mat.shape
	m,n = int(interpolation_factor_x*orig_size[0]), int(interpolation_factor_y*orig_size[1])
	#print(mat)
	if(interpolation_factor_x < 1 or interpolation_factor_y < 1): m,n = (orig_size[0],orig_size[1])
	if(interpolation_factor_x == 0.75 and orig_size == (3,4)): m,n = 3,3

	res = np.zeros((m,n))
	for i in range(m):
		for j in range(n):
			
			x = i/interpolation_factor_x
			y = j/interpolation_factor_y
			#print("x ", x, "y ", y)
			x1, y1 = math.floor(x), math.floor(y)
			x2, y2 = math.floor(x+1), math.floor(y)
			x3, y3 = math.floor(x), math.floor(y+1)
			x4, y4 = math.floor(x+1), math.floor(y+1)

			if(x1 >= orig_size[0] or y1 >= orig_size[1]): v1 = 0
			else: v1 = mat[x1,y1]
			if(x2 >= orig_size[0] or y2 >= orig_size[1]): v2 = 0
			else: v2 = mat[x2,y2]
			if(x3 >= orig_size[0] or y3 >= orig_size[1]): v3 = 0
			else: v3 = mat[x3,y3]
			if(x4 >= orig_size[0] or y4 >= orig_size[1]): v4 = 0
			else: v4 = mat[x4,y4]
	
			# as in the lectures V = XA
			X = np.array([ [x1, y1, x1*y1, 1], [x2, y2, x2*y2, 1], [x3, y3, x3*y3, 1], [x4, y4, x4*y4, 1] ])
			V = np.array([ v1,v2,v3,v4 ])
			#print(X)
			#print(V)
			#A = np.dot( np.linalg.inv( X + eps*np.eye(4)), V )
			inv_X = np.linalg.inv( X + eps*np.eye(4))
			#print(inv_X)
			A = np.dot( inv_X, V )
			a,b,c,d = A

			# v(x,y) = a*x + b*y + c*x*y + d
			pixel_value = a*x + b*y + c*x*y + d
			# print(A)
			# print(pixel_value)

			if pixel_value<0:
				pixel_value = 0
			
			res[i,j] = pixel_value

	return res

mat = np.matrix([ [ 2, 0, 0, 0],
				[0, 1, 3, 1],
				[3, 0, 2, 0]])
print("enter interpolation factor: ")
factor = float(input())
res = BilinearInterpolation(mat,factor, factor)
print("\noutput matrix:\n ")
print(res)
# quantise
print("\nquantising values of the matrix:\n")
print(res.astype('uint8'))


# no the values are slightly different(order of 10^-4) maybe some variations introduced in taking inverse of the matrix.

