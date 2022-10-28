import numpy as np


def recall_at_k(output, target, topk=(1,)):
    recall = {}
    correct = output == target
    for k in topk:
        correct_k = np.minimum(np.sum(correct[:, :k], axis=-1), 1)
        recall[k] = correct_k.sum() / len(output)
    return recall


def precision_at_k(output, target, topk=(1,)):
    recall = {}
    correct = output == target
    for k in topk:
        correct_k = np.sum(correct[:, :k], axis=-1) / k
        recall[k] = correct_k.sum() / len(output)
    return recall
