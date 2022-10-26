import torch
import numpy as np


class RetrievalBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, dataset, m, l):
        # m классов, в каждом по l картинок
        self.batch_size = m * l
        self.m, self.l = m, l

        self.__prepare_indices_dict(dataset)

    def __prepare_indices_dict(self, dataset):

        indices_dict = {}
        for _sample, _class in enumerate(dataset.labels):
            if indices_dict.get(_class) is None:
                indices_dict[_class] = []
            indices_dict[_class].append(_sample)

        self.indices = {}
        self.nrof_samples = 0
        for _class in indices_dict.keys():
            if len(indices_dict[_class]) < self.l:
                continue
            self.indices[_class] = indices_dict[_class]
            self.nrof_samples += len(self.indices[_class])
        self.classes = list(self.indices.keys())

    def __iter__(self):
        for _ in range(self.nrof_samples // self.batch_size):
            batch_classes = np.random.choice(self.classes, size=self.m, replace=False)
            out = []
            for cl in batch_classes:
                out.extend(np.random.choice(self.indices[cl], self.l, replace=False).tolist())
            yield out

    def __len__(self):
        return self.nrof_samples // self.batch_size
