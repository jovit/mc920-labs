# %%
# Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import subprocess
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'trab3'))
    print(os.getcwd())
except:
    pass

import cv2
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.util import invert
from skimage.morphology import binary_dilation, binary_erosion, binary_closing

in_file = "bitmap.pbm"
# out_file = sys.argv[2]

filename = os.path.join('./', in_file)

image = cv2.imread(filename, 0)
image = image / 255
image = image.astype(np.bool_)

image = invert(image)

# %% Step 1 - dilation
selem_1x100 = np.ones((1, 100))

dilated_image = binary_dilation(image, selem_1x100)

cv2.imwrite("step1.pbm", invert(dilated_image).astype(int))

# %% Step 2 - erosion
eroded_image = binary_erosion(dilated_image, selem_1x100)

cv2.imwrite("step2.pbm", invert(eroded_image).astype(int))

# %% Step 3 - dilation
selem_200x1 = np.ones((200, 1))

dilated_image2 = binary_dilation(image, selem_200x1)

cv2.imwrite("step3.pbm", invert(dilated_image2).astype(int))

# %% Step 4 - erosion
eroded_image2 = binary_erosion(dilated_image2, selem_200x1)

cv2.imwrite("step4.pbm", invert(eroded_image2).astype(int))

# %% Step 5 - resulting intersection
intersection = eroded_image * eroded_image2

cv2.imwrite("step5.pbm", invert(intersection).astype(int))

# %% Step 6 - closing
selem_1x30 = np.ones((1, 30))
closing_image = binary_closing(intersection, selem_1x30)
cv2.imwrite("step6.pbm", invert(closing_image).astype(int))

# %% Step 7 - Identify components


def draw_boxes(image, bounding_boxes):
    image_with_boxes = np.copy(image)
    for box in bounding_boxes:
        top_left = box[0]
        bottom_right = box[1]
        image_with_boxes[top_left[1], top_left[0]:bottom_right[0] + 1] = 0
        image_with_boxes[top_left[1] + 1, top_left[0]:bottom_right[0] + 1] = 0
        image_with_boxes[bottom_right[1], top_left[0]:bottom_right[0] + 1] = 0
        image_with_boxes[bottom_right[1] + 1,
                         top_left[0]:bottom_right[0] + 1] = 0
        image_with_boxes[top_left[1]:bottom_right[1] + 1, top_left[0]] = 0
        image_with_boxes[top_left[1]:bottom_right[1] + 1, top_left[0] + 1] = 0
        image_with_boxes[top_left[1]:bottom_right[1] + 1, bottom_right[0]] = 0
        image_with_boxes[top_left[1]                         :bottom_right[1] + 1, bottom_right[0]+1] = 0

    return image_with_boxes


f = open("./out.txt", "w")
subprocess.call(["gcc", "-o", "comp_conexos", "comp_conexos.c", "-lm"])
subprocess.call(["./comp_conexos", "step6.pbm", "step7.pbm"], stdout=f)
f.close()
f = open("./out.txt", "r")
out = f.read()
f.close()

out = out.split("\n")
out = out[4:]
bounding_boxes = []
for i in range(0, len(out) - 1, 2):
    splited_1 = out[i].split(",")
    splitted_2 = out[i+1].split(",")
    bounding_boxes.append(
        ((splited_1[0], splited_1[1]), (splitted_2[0], splitted_2[1])))
bounding_boxes = np.array(bounding_boxes).astype(int)

image_with_boxes = draw_boxes(invert(image).astype(int), bounding_boxes)
cv2.imwrite("step7_original.pbm", image_with_boxes)

# %% Step 8 - Calculating metrics


def relation_between_black_and_white(image, box):
    top_left = box[0]
    bottom_right = box[1]
    cropped = image[top_left[1]:bottom_right[1] +
                    1, top_left[0]:bottom_right[0] + 1]
    total = cropped.size
    black_pixels = total - np.count_nonzero(cropped)

    return black_pixels / total


def relation_between_transitions(image, box):
    top_left = box[0]
    bottom_right = box[1]
    horizontal_transitions = 0
    vertical_transitions = 0

    cropped = image[top_left[1]:bottom_right[1] +
                    1, top_left[0]:bottom_right[0] + 1]
    total = cropped.size

    for y, line in enumerate(cropped):
        for x, pixel in enumerate(line):
            # if black
            if pixel == 0:
                if y > 0 and cropped[y - 1, x] == 1:
                    vertical_transitions += 1
                if x > 0 and cropped[y, x - 1] == 1:
                    horizontal_transitions += 1
    return (horizontal_transitions/total, vertical_transitions/total)


original_image = invert(image).astype(int)
text_boxes = []

for box in bounding_boxes:
    black_and_white = relation_between_black_and_white(original_image, box)
    transitions = relation_between_transitions(original_image, box)

    if black_and_white < 0.5 and black_and_white > 0.2 and transitions[0] > 0.03 and transitions[0] < 0.1 and transitions[1] > 0.03 and transitions[1] < 0.1:
        text_boxes.append(box)

image_with_text_boxes = draw_boxes(invert(image).astype(int), text_boxes)
cv2.imwrite("step9.pbm", image_with_text_boxes)


# %% Step 10: Given the boxes with text, lets find the words
def separate_words(image, box):
    top_left = box[0]
    bottom_right = box[1]
    cropped = image[top_left[1]:bottom_right[1] +
                    1, top_left[0]:bottom_right[0] + 1]

    selem_1 = np.ones((6, 10))
    selem_2 = np.ones((10, 5))


    dilated1 = binary_dilation(cropped, selem_1)
    eroded1 = binary_dilation(dilated1, selem_1)

    dilated2 = binary_dilation(cropped, selem_2)
    eroded2 = binary_dilation(dilated2, selem_2)

    union = eroded1 * eroded2

    # io.imshow(cropped.astype(int), cmap="gray")
    # plt.show()

    # io.imshow(eroded1.astype(int), cmap="gray")
    # plt.show()

    # io.imshow(eroded2.astype(int), cmap="gray")
    # plt.show()

    # io.imshow(union.astype(int), cmap="gray")
    # plt.show()

    cv2.imwrite("temp.pbm", invert(union).astype(int))

    f = open("./out2.txt", "w")
    subprocess.call(["./comp_conexos", "temp.pbm", "temp2.pbm"], stdout=f)
    f.close()
    f = open("./out2.txt", "r")
    output = f.read()
    output = output.split("\n")
    output = output[4:]
    f.close()

    word_bounding_boxes = []
    for i in range(0, len(output) - 1, 2):
        splited_1 = output[i].split(",")
        splitted_2 = output[i+1].split(",")
        word_bounding_boxes.append(
            ((splited_1[0], splited_1[1]), (splitted_2[0], splitted_2[1])))
    word_bounding_boxes = np.array(word_bounding_boxes).astype(int)

    relative_boxes = []
    for b in word_bounding_boxes:
        b[0, 0] += top_left[0]
        b[0, 1] += top_left[1] 
        b[1, 0] += top_left[0]
        b[1, 1] += top_left[1] 

        relative_boxes.append(b)

    return relative_boxes

word_boxes = []
for box in text_boxes:
    current = separate_words(image, box)
    for b in current:
        word_boxes.append(b)

print("Number of words:", len(word_boxes))
cv2.imwrite("bla.pbm", invert(image).astype(int))
image_with_word_boxes = draw_boxes(invert(image).astype(int), word_boxes)
cv2.imwrite("step10.pbm", image_with_word_boxes)