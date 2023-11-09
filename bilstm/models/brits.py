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
        self.rits_b = rits.Model(self.input_size, self.seq_len, self.rnn_hid_size, self.aug, self.ext)

    def forward(self, data, mode="train"):
        ret_f = self.rits_f(data, 'forward', mode)
        ret_b = self.reverse(self.rits_b(data, 'backward', mode))
        ret = self.merge_ret(ret_f, ret_b)
        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

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

