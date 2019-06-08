# %%
# Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'trab4'))
    print(os.getcwd())
except:
    pass

import cv2
import sys
from skimage import io
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 4

in_file1 = sys.argv[1]
in_file2 = sys.argv[2]

distance_ratio = sys.argv[3]
distance_ratio = float(distance_ratio)

filename1 = os.path.join('./', in_file1)
filename2 = os.path.join('./', in_file2)

image1_ = cv2.imread(filename1).astype(np.uint8)
image1 = cv2.cvtColor(image1_, cv2.COLOR_BGR2GRAY).astype(np.uint8)

image2_ = cv2.imread(filename2).astype(np.uint8)
image2 = cv2.cvtColor(image2_, cv2.COLOR_BGR2GRAY).astype(np.uint8)


def get_sift_descriptors(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    return kp, des


def match_sift_descriptors(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    return matches


def join_images_by_matches(img1_, img2_, kp1, kp2, des1, des2, matches):
    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < distance_ratio*m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    image_with_matches = cv2.drawMatchesKnn(
        img1_, kp1, img2_, kp2, matches, None)
    cv2.imwrite('matches.jpg', image_with_matches)
    if len(matches) > 0 and len(matches[:, 0]) >= MIN_MATCH_COUNT:
        src = np.float32(
            [kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32(
            [kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        try:
            dst = cv2.warpPerspective(
                img1_, H, (img2_.shape[1] + img1_.shape[1], img2_.shape[0]))

            dst[0:img2_.shape[0], 0:img2_.shape[1]] = img2_
            return image_with_matches, dst
        except:
            raise AssertionError("Can't find enough keypoints.")
    else:
        raise AssertionError("Can't find enough keypoints.")


kp1, des1 = get_sift_descriptors(image1)
kp2, des2 = get_sift_descriptors(image2)

matches = match_sift_descriptors(des1, des2)

image_with_matches, joined = join_images_by_matches(image1_, image2_, kp1, kp2, des1, des2, matches)
cv2.imwrite('sift_result/matches.jpg', image_with_matches)
cv2.imwrite('sift_result/joined.jpg', joined)
