import cv2
import functools
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

from os import path
from sys import argv
from time import time
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


def normalize(image):
    return (image / 255).reshape(-1, 3)

def cluster(image, n_clusters, n_init=10, max_iter=300):
    normalized = normalize(image)
    k_colors = KMeans(n_clusters, n_init=n_init, max_iter=max_iter).fit(normalized)
    compressed = k_colors.cluster_centers_[k_colors.labels_]
    compressed = np.reshape(compressed, (image.shape))
    return compressed, k_colors.labels_, k_colors.cluster_centers_

def plot_clusters(image, labels, colors, show=True, save_fname=None):
    r = image[:, :, 0].flatten()
    g = image[:, :, 1].flatten()
    b = image[:, :, 2].flatten()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r, g, b, c=colors[labels] / 255)

    if save_fname: plt.savefig(save_fname, bbox_inches='tight')
    if show: plt.show()

def plot_histogram(n_clusters, colors, show=True, save_fname=None):
    hist, _ = np.histogram(labels, bins=n_clusters)
    hist = hist.astype("float") / hist.sum()
    
    colors = colors[(-hist).argsort()]
    hist = hist[(-hist).argsort()]
    
    chart = np.zeros((50, 500, 3), dtype=np.uint8)

    start = 0
    for i in range(n_clusters):
        end = start + hist[i] * 500
        r, g, b = colors[i][0:3]
        cv2.rectangle(img=chart, pt1=(int(start), 0), pt2=(int(end), 50), color=(r, g, b), thickness=-1)
        start = end	
    
    plt.figure()
    plt.axis("off")
    plt.imshow(chart)
    if save_fname: plt.savefig(save_fname, bbox_inches='tight')
    if show: plt.show()

try:
    image_fname = argv[1]
    image_name, image_ext = path.splitext(path.split(image_fname)[-1])
    n_clusters = int(argv[2]) if len(argv) > 2 else 128
    save_path = argv[3] if len(argv) > 3 else ""
except:
    print('\nusage: k_means.py image_fname [n_clusters] [save_path]')
    exit()

image = cv2.cvtColor(cv2.imread(image_fname), cv2.COLOR_BGR2RGB)
k_image, labels, cluster_centers = cluster(image, n_clusters)

fname = path.join(save_path, f"{image_name}{n_clusters}{image_ext}")
img.imsave(fname, k_image)
print(f"Image saved to {fname}")

# denormalizes values
k_image *= 255
cluster_centers = cluster_centers * 255

plot_clusters(image, labels, colors=cluster_centers)

plot_histogram(n_clusters, colors=cluster_centers)