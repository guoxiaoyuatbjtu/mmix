import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from models.astgcn import make_model 

import math
import random
import numpy as np

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


class Model(nn.Module):

    def __init__(self, input_size, seq_len, rnn_hid_size, aug, ext):
        super(Model, self).__init__()
        self.input_size   = input_size
        self.seq_len      = seq_len
        self.rnn_hid_size = rnn_hid_size
        self.aug          = aug
        self.ext          = ext
        self.build()

    def get_adjacency_matrix(self, distance_df_filename, num_of_vertices, id_filename=None):
        '''
        Parameters
        ----------
        distance_df_filename: str, path of the csv file contains edges information

        num_of_vertices: int, the number of vertices

        Returns
        ----------
        A: np.ndarray, adjacency matrix

        '''
        if 'npy' in distance_df_filename:

            adj_mx = np.load(distance_df_filename)

            return adj_mx, None

        else:

            import csv

            A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                         dtype=np.float32)

            distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                                dtype=np.float32)

            if id_filename:

                with open(id_filename, 'r') as f:
                    id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

                with open(distance_df_filename, 'r') as f:
                    f.readline()
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) != 3:
                            continue
                        i, j, distance = int(row[0]), int(row[1]), float(row[2])
                        A[id_dict[i], id_dict[j]] = 1
                        distaneA[id_dict[i], id_dict[j]] = distance
                return A, distaneA

            else:

                with open(distance_df_filename, 'r') as f:
                    f.readline()
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) != 3:
                            continue
                        i, j, distance = int(row[0]), int(row[1]), float(row[2])
                        A[i, j] = 1
                        distaneA[i, j] = distance
                return A, distaneA

    def build(self):
        self.adj_mx, _ = self.get_adjacency_matrix('json/PEMS08.csv', 170)
        self.net = make_model(torch.device('cuda:0'), 2, 1, 3, 64, 64, 4, self.adj_mx, 48, 48, 170)

    def forward(self, data, direct, mode="train"):

        values  = data[direct]['values']
        masks   = data[direct]['masks']

        if mode == "train":
            if self.aug == 'mmix':
                mixed_masks = self.mmix_aug(masks, self.ext['k'])
            if self.aug == 'none':
                mixed_masks = masks
            if self.aug == 'unif':
                mixed_masks = self.unif_aug(masks, self.ext['rate'])
            if self.aug == 'gaus':
                mixed_masks = masks
            if self.aug == 'temp':
                mixed_masks = self.temp_aug(masks)
            if self.aug == 'spat':
                mixed_masks = self.spat_aug(masks)
            if self.aug == 'spat_temp':
                mixed_masks = self.spat_temp_aug(masks)
        else:
            mixed_masks = masks
        
        # Original data
        x = values
        m = mixed_masks

        inputs = (x * m).transpose(1, 2).unsqueeze(2)

        outputs = self.net(inputs).transpose(1, 2)

        loss_a = torch.sum(torch.abs(values - outputs) * m, dim=(1,2)) / (torch.sum(m, dim=(1,2)) + 1e-5)  / (masks.sum() + 1e-5) * m.sum()
        loss_b = torch.sum(torch.abs(values - outputs) * (masks - m), dim=(1,2)) / (torch.sum((masks - m), dim=(1,2)) + 1e-5) / (masks.sum() + 1e-5) * (masks - m).sum()
        loss = loss_a.mean() + loss_b.mean()

        imputations = masks * values + (1 - masks) * outputs

        return {'loss': loss, 'imputations': imputations, 'values': values, 'masks': masks}

    def mmix_aug(self, masks, k):
        nums = masks.shape[1] // k
        if masks.shape[1] % k != 0:
            print('Sequence length should be divided by k with no remainder')
            exit()
        fragments = []
        for _ in range(nums):
            fragment_start = random.randint(0, self.ext['seed'].shape[0] - masks.shape[1])
            fragments.append(self.ext['seed'][fragment_start:fragment_start+k])
        masks_mix = torch.cat(fragments, dim=0)
        public_masks = masks_mix * masks
        return public_masks
    
    def unif_aug(self, masks, rate=0.1):
        masks = masks * (torch.rand(masks.shape).cuda() > rate)
        return masks

    def gaus_aug(self, x):
        return x + 0.1 * x * torch.randn(x.shape).cuda()

    def temp_aug(self, masks):
        temp_masks_seed = (torch.rand(masks.shape).cuda() > 0.033).int()
        # Mask by Time Offset
        # Length = 3
        temp_masks = temp_masks_seed
        temp_masks[:,1:] = temp_masks[:,1:] * temp_masks_seed[:,:-1]
        temp_masks[:,2:] = temp_masks[:,2:] * temp_masks_seed[:,1:-1]
        masks = temp_masks * masks
        return masks

    def spat_aug(self, masks):
        spat_masks_seed = (torch.rand(masks.shape).cuda() > 0.033).int()
        spat_masks = spat_masks_seed
        spat_masks[:,:,1:] = spat_masks[:,:,1:] * spat_masks_seed[:,:,:-1]
        spat_masks[:,:,:-1] = spat_masks[:,:,:-1] * spat_masks_seed[:,:,1:]
        masks = spat_masks * masks
        return masks

    def spat_temp_aug(self, masks):
        spat_masks_seed = (torch.rand(masks.shape).cuda() > 0.011).int()
        spat_masks = spat_masks_seed
        spat_masks[:,:,1:] = spat_masks[:,:,1:] * spat_masks_seed[:,:,:-1]
        spat_masks[:,:,:-1] = spat_masks[:,:,:-1] * spat_masks_seed[:,:,1:]
        temp_masks_seed = spat_masks_seed
        temp_masks = temp_masks_seed
        temp_masks[:,1:] = temp_masks[:,1:] * temp_masks_seed[:,:-1]
        temp_masks[:,2:] = temp_masks[:,2:] * temp_masks_seed[:,1:-1]
        masks = temp_masks * masks
        return masks

    def parse_delta(self, masks):
        deltas = []
        for time_step in range(self.seq_len):
            if time_step == 0:
                deltas.append(torch.ones(masks.shape[0], 1, masks.shape[2]).cuda())
            else:
                deltas.append(torch.ones(masks.shape[0], 1, masks.shape[2]).cuda() + (1 - masks[:,time_step:(time_step+1)]) * deltas[-1])
        deltas = torch.cat(deltas, dim=1)
        return deltas
