from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np


def prepare_img(tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inv_normalize(tensor).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def plot_topn_with_labels(queries, query_labels, retrieval_set, retrieval_labels, k=5, title=None):
    n = len(queries)
    _, axs = plt.subplots(n, 1, figsize=(18, 10))
    h, w = 50, 50
    for i in range(n):
        images = []
        for j in range(k + 1):
            if j == 0:
                images.append(cv2.resize(queries[i], (h, w)))
                continue
            border_color = [255, 0, 0] if query_labels[i] != retrieval_labels[i][j - 1] else [0, 255, 0]
            img = cv2.copyMakeBorder(retrieval_set[i][j - 1], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border_color)
            images.append(cv2.resize(img, (h, w)))
        axs[i].imshow(np.hstack(images).astype(np.uint8))
        axs[i].axis('off')
    if title is not None:
        plt.suptitle(title)
    plt.show()


def plot_topn(queries, retrieval_set, k=5, title=None):
    n = len(queries)
    _, axs = plt.subplots(n, 1, figsize=(18, 10))
    h, w = 50, 50
    for i in range(n):
        images = []
        for j in range(k + 1):
            if j == 0:
                images.append(cv2.resize(queries[i], (h, w)))
                continue
            images.append(cv2.resize(retrieval_set[i][j - 1], (h, w)))
        axs[i].imshow(np.hstack(images).astype(np.uint8))
        axs[i].axis('off')
    if title is not None:
        plt.suptitle(title)
    plt.show()
