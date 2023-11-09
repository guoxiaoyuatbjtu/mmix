import torch
import torch.nn as nn

from torch.autograd import Variable

from models import rits


class Model(nn.Module):
    def __init__(self, input_size, seq_len, rnn_hid_size, aug, ext):
        super(Model, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.input_size   = input_size
        self.seq_len      = seq_len
        self.aug          = aug
        self.ext          = ext
        self.build()

    def build(self):
        self.rits_f = rits.Model(self.input_size, self.seq_len, self.rnn_hid_size, self.aug, self.ext)

    def forward(self, data, mode="train"):
        ret_f = self.rits_f(data, 'forward', mode)
        return ret_f

    def run_on_batch(self, data, optimizer, epoch=None):

        if optimizer is not None:
            ret = self(data, mode="train")
            optimizer.zero_grad()
            ret['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 15, norm_type=2)
            optimizer.step()
        else:
            ret = self(data, mode="eval")

        return ret

