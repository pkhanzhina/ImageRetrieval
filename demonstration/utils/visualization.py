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
    # return (inv_normalize(tensor).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def plot_topn_old(queries, retrieval_set, k=5, title=None):
    n = len(queries)
    _, axs = plt.subplots(n, k + 1, figsize=(18, 10))
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


def plot_topn(queries, retrieval_set, distances=None, labels=None, k=5, title=None):
    n = len(queries)
    _, axs = plt.subplots(n, k + 1, figsize=(12, 7))
    h, w = 224, 224
    for i in range(n):
        axs[i, 0].imshow(cv2.resize(queries[i], (h, w)))
        query_labels = None
        if labels is not None:
            axs[i, 0].set_title(f"class - {labels[0][i]}")
            query_labels = labels[0][i]
        for j in range(k):
            if labels is not None:
                border_color = [255, 0, 0] if query_labels != labels[1][i][j] else [0, 255, 0]
                img = cv2.copyMakeBorder(retrieval_set[i][j], 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=border_color)
            else:
                img = retrieval_set[i][j]
            axs[i, j + 1].imshow(cv2.resize(img, (h, w)).astype(np.uint8))
            subtitle = ''
            if labels is not None:
                subtitle = f"class - {labels[1][i][j]}"
            if distances is not None:
                subtitle += "\ndist - {:.4f}".format(distances[i][j])
            if len(subtitle):
                axs[i, j + 1].set_title(subtitle)

    for i in range(n):
        for j in range(k + 1):
            axs[i, j].axis('off')
    if title is not None:
        plt.suptitle(title)
    plt.show()
