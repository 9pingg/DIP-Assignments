import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
RUN = input("enter choice Q1, Q2, Q3: ")

def get_rotation(in_deg):
    a = np.radians(in_deg)
    return np.array([
        [ np.cos(a), -np.sin(a), 0 ],
        [ np.sin(a), np.cos(a), 0 ],
        [ 0, 0, 1 ]
])

def get_scaling(s_x, s_y):
    return np.array([
        [s_x,0,0],
        [0,s_y,0],
        [0,0,1]
    ])

def get_translation(t_x, t_y):
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [t_x, t_y, 1]
    ])

# function to apply geometric transformation to an image. takes in image img, Transformation matrix T and output_dim as parameters.
# returns transformed image
def GeometricTransformation(img, T, output_dim=300, eps=1e-4, for_reg=False):
    orig_size = img.shape
    if(for_reg==False):
        output_dim = int(1.5 * img.shape[0])
    else:
        output_dim = orig_size[0]
    #print(output_dim)
    img1 = np.zeros((output_dim, output_dim))
    T_inv = np.linalg.inv(T)
    for i in range(output_dim):
        for j in range(output_dim):
            X = np.array([i, j, 1])
            V = np.dot(X, T_inv)
            x,y = V[0],V[1]
            x1, y1 = math.floor(x), math.floor(y)
            x2, y2 = math.floor(x+1), math.floor(y)
            x3, y3 = math.floor(x), math.floor(y+1)
            x4, y4 = math.floor(x+1), math.floor(y+1)

            if(x1 >= orig_size[0] or y1 >= orig_size[1] or x1 < 0 or y1 < 0): v1 = 0
            else: v1 = img[x1,y1]
            if(x2 >= orig_size[0] or y2 >= orig_size[1] or x2 < 0 or y2 < 0): v2 = 0
            else: v2 = img[x2,y2]
            if(x3 >= orig_size[0] or y3 >= orig_size[1] or x3 < 0 or y3 < 0): v3 = 0
            else: v3 = img[x3,y3]
            if(x4 >= orig_size[0] or y4 >= orig_size[1] or x4 < 0 or y4 < 0): v4 = 0
            else: v4 = img[x4,y4]
            # if(i == 0 and j == 1): 
            #         print(v1,v2,v3,v4)
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

            if pixel_value < 0 or pixel_value > 255:
                pixel_value = 0
            if(pixel_value > 0):
                print("output grid coords : ", i , j, "matches to ", x, y, "pixel_value: ", pixel_value)
            img1[i,j] = pixel_value
    return img1.astype("uint8")

# function to register an image
def registeration(img):
    # As we already know the T matrix there is no need to calculate T using (U'U)-1 U'X    
    T = np.dot(get_translation(50,50), get_rotation(10))
    T_inv = np.linalg.inv(T)
    return GeometricTransformation(img, T_inv, for_reg = True)

# function to plot normalized histograms
def plot_normalized_histogram(img, legend, L=256):
    freq = np.zeros(L)
    img_size = img.shape
    
    # calc histogram
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            pixel_value = img[i,j]
            freq[pixel_value] = freq[pixel_value] + 1
    
    # normalize
    freq = freq/(img_size[0]*img_size[1])
    plt.plot(list(range(L)), freq, ":", label = legend)


# function to find the log transformation of an image
def log_transformation(img, L=256):
    c = (L-1) / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    log_image = np.array(log_image, dtype = np.uint8)
    return log_image

# function to match histograms 
def histogram_matching(img, log_img, L=256):    
    # histogram
    h = np.zeros(L)
    g = np.zeros(L)
    H = np.zeros(L)
    G = np.zeros(L)
    img_size = img.shape
    match_img = np.zeros(img_size)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            h_pixel_val = img[i,j]
            g_pixel_val = log_img[i,j]
            h[h_pixel_val] = h[h_pixel_val] + 1
            g[g_pixel_val] = g[g_pixel_val] + 1
    
   # normalize
    h = h/(img_size[0]*img_size[1])
    g = g/(img_size[0]*img_size[1])
    
    # cdf    
    for i in range(L):
        if(i == 0):
            H[0] = h[0]
            G[0] = g[0]
            continue
        H[i] = H[i-1] + h[i]
        G[i] = G[i-1] + g[i]
    # transform
    match_transform = []
    for r in range(L):
        min_diff = sys.maxsize
        level_pointer = r
        for s in range(L):
            diff = abs(H[r] - G[s])
            if diff < min_diff:
                min_diff = diff
                level_pointer = s
        match_transform.append(level_pointer) # position r points to s
    #plt.plot(match_transform, label="log transform")

    # output
    for level in range(L):
        match_img[np.where(img==level)] = match_transform[level]
    
    return match_img.astype("uint8")

if RUN == "Q1":
    T = np.dot(get_translation(50,50), get_rotation(10))
    img = cv2.imread("x5_size.bmp", cv2.IMREAD_GRAYSCALE)
    img1 = GeometricTransformation( img, T)
    print("joint transformation for translation of (50,50) followed by rotation of 10 degree. \nT: ")
    print(T)
    cv2.imshow("original_image", img)
    cv2.imshow("transformed_image", img1)
    cv2.imwrite("q1output.png", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if RUN == "Q2":
    img_orig = cv2.imread("x5_size.bmp", cv2.IMREAD_GRAYSCALE)
    img_unreg = cv2.imread("q1output.png", cv2.IMREAD_GRAYSCALE)
    
    img_reg = registeration(img_unreg)
    
    cv2.imshow("original", img_orig)
    cv2.imshow("unregistered", img_unreg)
    cv2.imshow("registered", img_reg)
    cv2.imwrite("q2output.png", img_reg)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if RUN == "Q3":
    img = cv2.imread("x5.bmp", cv2.IMREAD_GRAYSCALE)
    log_img = log_transformation(img)
    match_img = histogram_matching(img, log_img)
    plot_normalized_histogram(img, "Input Image")
    plot_normalized_histogram(log_img, "Log_transformed Image")
    plot_normalized_histogram(match_img, "Matched Image")
    plt.title("Normalized Histograms")
    plt.legend()
    plt.show()
    
    cv2.imshow("Input Image", img)
    cv2.imshow("Log_transformed Image", log_img)
    cv2.imshow("Matched Image", match_img)
    cv2.imwrite("q3log_img.png", log_img)
    cv2.imwrite("q3matched_img.png", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()













