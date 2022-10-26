import os
import pandas
import pickle

import pandas as pd
import torch.utils.data as data
from PIL import Image


class RetrievalDataset(data.Dataset):
    def __init__(self, root_dir, data_type, transforms=None):
        self.root_dir = root_dir
        self.data_type = data_type
        self.transforms = transforms

        with open(os.path.join(root_dir, f'{data_type}_classes.pickle'), 'rb') as f:
            class_indices = pickle.load(f)
        annot_df = pd.read_csv(os.path.join(root_dir, 'annotation.csv'))
        annot_df = annot_df[annot_df['label'].isin(class_indices)]
        self.paths = list(annot_df['path'].values)
        self.labels = list(annot_df['label'].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.paths[idx])).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.labels[idx]