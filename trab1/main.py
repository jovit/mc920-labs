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

#%%
# Take the 2-dimensional DFT and centre the frequencies
ftimage = np.fft.fft2(image)
ftimage = np.fft.fftshift(ftimage)
magnitude_spectrum = 20*np.log(np.abs(ftimage))
io.imshow(np.abs(magnitude_spectrum), cmap='gray')
plt.show()


# Build and apply a Gaussian filter.
frequencies = [10, 20, 30, 40, 50, 60]
for d0 in frequencies:
    nrows = len(image)
    ncols = len(image[0])
    cy, cx = nrows/2, ncols/2
    # Generate the values for u, v
    x = np.arange(nrows)
    y = np.arange(ncols)
    # create a 2D array for the value of u and v for each point
    X, Y = np.meshgrid(x, y)
    # make an gaussian mask, we subtract X and Y by c, because our fourrier is shifted
    # so the origin is no longer in (0,0), so we need to shift the mask to the center
    gmask = np.exp(
        (-((X-cx) ** 2) / (2 * (d0 ** 2))) +
        (-((Y-cy) ** 2) / (2 * (d0 ** 2) ))
    )
    
    ftimagep = ftimage * gmask # apply tha mask
    magnitude_spectrum = 20 * np.log(np.abs(ftimagep)) # generate a plottable spectrum
    io.imshow(np.abs(magnitude_spectrum), cmap='gray')
    plt.show()

    # Finally, take the inverse transform and show the blurred image
    imagep = np.fft.ifft2(ftimagep)
    io.imshow(np.abs(imagep), cmap='gray')
    plt.show()