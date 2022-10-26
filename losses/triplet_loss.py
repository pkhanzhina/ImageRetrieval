import torch.nn as nn
import numpy as np
import torch


class TripletLoss(nn.Module):
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        super().__init__()

    def all_combinations(self, ind):
        anchors, positives = torch.meshgrid(ind, ind)
        return anchors.flatten(), positives.flatten()

    def choose_triplets(self, pairwise_dist, target):
        anchor, positive, negative = [], [], []
        for label in torch.unique(target):
            same_indices = torch.where(target == label)[0]
            negative_indices = torch.where(target != label)[0]
            anchors, positives = self.all_combinations(same_indices)
            for a, p in zip(anchors, positives):
                if a == p:
                    continue
                loss = pairwise_dist[a, p] - pairwise_dist[a, negative_indices] + self.alpha
                candidates = negative_indices[torch.logical_and(0 < loss, loss < self.alpha)]
                if candidates.size(0) == 0:
                    continue
                semi_hard = np.random.choice(candidates)
                anchor.append(a), positive.append(p), negative.append(torch.tensor(semi_hard))
        return torch.stack(anchor), torch.stack(positive), torch.stack(negative)

    def forward(self, output, target):
        pairwise_dist = 2 - 2 * (output @ output.transpose(0, 1))
        anchor, positive, negative = self.choose_triplets(pairwise_dist, target)

        loss = torch.maximum(
            pairwise_dist[anchor, positive] - pairwise_dist[anchor, negative] + self.alpha, torch.tensor(0)).sum()
        return loss