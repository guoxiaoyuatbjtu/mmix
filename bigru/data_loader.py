import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, datapath):
        super(MySet, self).__init__()
        self.content = open(datapath).readlines()

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        return rec

def collate_fn(recs):
    forward  = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))
        masks  = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))
        return {'values': values, 'masks': masks, 'deltas': deltas}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    return ret_dict

def get_loader(datapath, batch_size = 32, shuffle = True):
    data_set = MySet(datapath)
    data_iter = DataLoader(dataset = data_set, batch_size = batch_size, num_workers = 4, shuffle = shuffle, pin_memory = True, collate_fn = collate_fn)
    return data_iter
