import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import random


class Model(nn.Module):
    def __init__(self, input_size, seq_len, rnn_hid_size, aug, ext):
        super(Model, self).__init__()
        self.input_size   = input_size
        self.seq_len      = seq_len
        self.rnn_hid_size = rnn_hid_size
        self.aug          = aug
        self.ext          = ext
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.input_size * 2, self.rnn_hid_size)
        self.fc = nn.Sequential(
            nn.Linear(self.rnn_hid_size, self.rnn_hid_size),
            nn.ReLU(),
            nn.Linear(self.rnn_hid_size, self.rnn_hid_size),
            nn.ReLU(),
            nn.Linear(self.rnn_hid_size, self.input_size))

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

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        
        x_loss_a = 0.0
        x_loss_b = 0.0

        imputations = []

        for t in range(self.seq_len):

            # Original data
            x = values[:, t, :]

            if self.aug == 'gaus' and mode == 'train':
                x = self.gaus_aug(x)

            m = mixed_masks[:, t, :]

            inputs = torch.cat([x * m, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            outputs = self.fc(h)

            x_loss_a += torch.sum(torch.abs(values[:,t,:] - outputs) * m[:,t,:], dim=1)
            x_loss_b += torch.sum(torch.abs(values[:,t,:] - outputs) * (masks - m)[:,t,:], dim=1)

            imputed = masks[:,t,:] * x + (1 - masks[:,t,:]) * outputs

            imputations.append(imputed.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        x_loss  = x_loss_a / (torch.sum(m, dim=(1,2)) + 1e-5) / (torch.sum(masks, dim=(1,2)) + 1e-5) * torch.sum(m, dim=(1,2))
        x_loss += x_loss_b / (torch.sum((masks - m), dim=(1,2)) + 1e-5) / (torch.sum(masks, dim=(1,2)) + 1e-5) * torch.sum((masks - m), dim=(1,2))
        x_loss  = x_loss.mean()

        return {'loss': x_loss, 'imputations': imputations, 'values': values, 'masks': masks}

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
