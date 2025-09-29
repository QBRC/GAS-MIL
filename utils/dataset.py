import torch
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data._utils.collate import default_collate
import numpy as np

class Dataset(BaseDataset):
    def __init__(self, df, keys, max_tile=200):
        self.df = df
        self.max_tile = max_tile
        self.keys = keys
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        features = [np.load(self.df[key][idx]) for key in self.keys]
        ns = [f.shape[0] for f in features]
        max_n = self.max_tile
        for i, n in enumerate(ns):
            if n < max_n:
                padding_features = np.zeros((max_n-n, features[i].shape[1]))
                features[i] = np.concatenate((features[i], padding_features), axis=0)
            else:
                features[i] = features[i][:max_n]
        feature = np.concatenate(features, axis=1)
        feature = feature.astype(np.float32)
        feature = torch.tensor(feature)
        label = self.df['label'][idx]
        image_id = self.df['image_id'][idx]

        return feature, label, image_id


def pad_collate_fn(batch,
                   batch_first=True,
                   max_len=None):
    sequences = [item[0] for item in batch]
    others = [item[1:] for item in batch]

    if max_len is None:
        max_len = max([s.size(0) for s in sequences])

    trailing_dims = sequences[0].size()[1:]

    if batch_first:
        padded_dims = (len(sequences), max_len) + trailing_dims
        masks_dims = (len(sequences), max_len, 1)
    else:
        padded_dims = (max_len, len(sequences)) + trailing_dims
        masks_dims = (max_len, len(sequences), 1)

    padded_sequences = sequences[0].data.new(*padded_dims).fill_(0.0)
    masks = torch.ones(*masks_dims, dtype=torch.bool)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            padded_sequences[i, :length, ...] = tensor[:max_len, ...]
            masks[i, :length, ...] = False
        else:
            padded_sequences[:length, i, ...] = tensor[:max_len, ...]
            masks[:length, i, ...] = False

    others = default_collate(others)

    return padded_sequences, masks, *others