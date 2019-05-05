#%%
#Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'trab2'))
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

filename = os.path.join('./', 'still_alive.jpg')

image = cv2.imread(filename, 0)
image = image.astype(float)

io.imshow(image.astype(int), cmap='gray')
plt.show()

# The Laplacian of Gaussian
ordered_dithering = np.array([
    [6., 8., 4.],
    [1., 0., 3,],
    [5., 2., 7.]
]).astype(float)


bayer_ordered_dithering = np.array([
    [0., 12., 3., 15.],
    [8., 4., 11., 7.],
    [2., 14., 1., 13.],
    [10., 6., 9., 5.],
]).astype(float)

def normalize_to_interval(gmin, gmax, image):
    fmax = np.amax(image)
    fmin = np.amin(image)
    a = (gmax - gmin)/(fmax - fmin)
    normalized = np.round((image - fmin) * a + gmin) 

    return normalized

def apply_ordered_dithering(image, pattern):
    pattern_size = len(pattern)
    normalized = normalize_to_interval(0., 9., image)

    for i, line in enumerate(normalized):
        for j, pixel in enumerate(line):
            if pixel > pattern[i % pattern_size , j % pattern_size]:
                normalized[i, j] = 9
            else:
                normalized[i, j] = 0
    return normalize_to_interval(0, 255, normalized)

# left to right pattern
def apply_dithering_with_error_diffusion(image):
    image_height = len(image)
    image_width = len(image[0])
    image_copy = np.copy(image)
    for i, line in enumerate(image_copy):
        for j, pixel in enumerate(line):
            error = 0
            if pixel > 128:
                error = pixel - 255
                image_copy[i, j] = 255
            else:
                error = pixel
                image_copy[i, j] = 0
            if j + 1 < image_width - 1:
                image_copy[i, j + 1] += 7./16. * error
            if i + 1 < image_height - 1 and j - 1 > 0:
                image_copy[i + 1, j - 1] += 3./16. * error
            if i + 1 < image_height - 1:
                image_copy[i + 1, j] += 5./16. * error
            if i + 1 < image_height - 1 and j + 1 < image_width - 1:
                image_copy[i + 1, j + 1] += 1./16. * error
    return image_copy

# alternating pattern
def apply_alternating_dithering_with_error_diffusion(image):
    image_height = len(image)
    image_width = len(image[0])
    image_copy = np.copy(image)
    for i, line in enumerate(image_copy):
        order = enumerate(line)
        going_right = True
        if i % 2 != 0:
            order = list(order)[::-1]
            going_right = False

        for j, pixel in order:
            error = 0
            if pixel > 128:
                error = pixel - 255
                image_copy[i, j] = 255
            else:
                error = pixel
                image_copy[i, j] = 0
            if j + 1 < image_width - 1 and going_right:
                image_copy[i, j + 1] += 7./16. * error

            if j - 1 > 0 and not going_right:
                image_copy[i, j - 1] += 7./16. * error

            if i + 1 < image_height - 1 and j - 1 > 0:
                image_copy[i + 1, j - 1] += 3./16. * error
            if i + 1 < image_height - 1:
                image_copy[i + 1, j] += 5./16. * error
            if i + 1 < image_height - 1 and j + 1 < image_width - 1:
                image_copy[i + 1, j + 1] += 1./16. * error
    return image_copy

#%%
dithered_image = apply_ordered_dithering(image, bayer_ordered_dithering)
io.imshow(dithered_image.astype(int), cmap='gray')
plt.show()

#%%
error_diffusion = apply_dithering_with_error_diffusion(image)
io.imshow(error_diffusion.astype(int), cmap='gray')
plt.show()
#%%
alternating_error_diffusion = apply_alternating_dithering_with_error_diffusion(image)
io.imshow(alternating_error_diffusion.astype(int), cmap='gray')
plt.show()

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
