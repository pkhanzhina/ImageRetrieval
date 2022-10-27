from .base import *
import pickle
import pandas as pd


class Flowers(BaseDataset):
    def __init__(self, root, classes, transform=None):
        BaseDataset.__init__(self, root, classes, transform)
        # amount of images deviates slightly from what's reported online

        if classes == range(0, 169):
            data_type = 'train'
        elif classes == range(169, 169 + 42):
            data_type = 'test'
        else:
            print('Unknown range for classes selected')
            input()

        with open(os.path.join(root, f'{data_type}_classes.pickle'), 'rb') as f:
            class_indices = pickle.load(f)

        annot_df = pd.read_csv(os.path.join(root, 'annotation.csv'))
        annot_df = annot_df[annot_df['label'].isin(class_indices)]

        for i, l in annot_df.iterrows():
            self.im_paths += [os.path.join(root, l['path'])]
            self.ys += [int(l['label'])]
            self.I += [i]

        idx_to_class = {idx: i for i, idx in enumerate(
            sorted(set(self.ys))
        )}
        self.ys = list(
            map(lambda x: idx_to_class[x], self.ys))

    def nb_classes(self):
        assert len(set(self.ys)) == len(set(self.classes))
        return len(self.classes)

