#%%
#Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'trab1'))
	print(os.getcwd())
except:
	pass

import cv2
from skimage import io, data_dir
from skimage import data
from skimage import img_as_float
from scipy.ndimage.filters import convolve
from matplotlib import pyplot as plt
import math
import numpy as np

filename = os.path.join('./', 'butterfly.png')

image = io.imread(filename)
image = image.astype(float)

io.imshow(image.astype(int), cmap='gray')
plt.show()

# The Laplacian of Gaussian
filter_h1 = np.array([
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0]
]).astype(float)

filter_h2 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 4],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]).astype(float) / 256

filter_h3 = np.array([
    [-1, 0 , 1],
    [-2, 0 , 2],
    [-1, 0, 1]
]).astype(float)

filter_h4 = np.array([
    [-1, -2 , -1],
    [0, 0 , 0],
    [1, 2, 1]
]).astype(float)

#%%
image_h1= cv2.filter2D(image, -1, filter_h1)
cv2.normalize(image_h1, image_h1, 0, 255, cv2.NORM_MINMAX)
io.imshow(image_h1.astype(int), cmap='gray')
plt.show()

#%%
image_h2 = cv2.filter2D(image, -1 ,filter_h2)
cv2.normalize(image_h2, image_h2, 0, 255, cv2.NORM_MINMAX)
io.imshow(image_h2.astype(int), cmap='gray')
plt.show()

#%%
image_h3 = cv2.filter2D(image, -1, filter_h3)
image_h3_normalized = np.copy(image_h3)
cv2.normalize(image_h3, image_h3_normalized, 0, 255, cv2.NORM_MINMAX)
io.imshow(image_h3_normalized.astype(int), cmap='gray')
plt.show()

#%%
image_h4 = cv2.filter2D(image, -1, filter_h4)
image_h4_normalized = np.copy(image_h4)
cv2.normalize(image_h4, image_h4_normalized, 0, 255, cv2.NORM_MINMAX)
io.imshow(image_h4_normalized.astype(int), cmap='gray')
plt.show()

#%%
#Sobel operator
composed = np.sqrt(np.power(image_h3.astype(float), 2) + np.power(image_h4.astype(float), 2))
cv2.normalize(composed, composed, 0, 255, cv2.NORM_MINMAX)
io.imshow(composed.astype(int), cmap='gray')
plt.show()