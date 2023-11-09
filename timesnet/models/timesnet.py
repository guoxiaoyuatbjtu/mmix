import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from models.Embed import DataEmbedding
from models.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len=48, k=2, d_model=64, d_ff=64, num_kernels=6):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.k = k 
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(self.d_model, self.d_ff,
                               num_kernels=self.num_kernels),
            nn.GELU(),
            Inception_Block_V1(self.d_ff, self.d_model,
                               num_kernels=self.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len) % period != 0:
                length = (
                                 ((self.seq_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, enc_in=1, c_out=1, seq_len=48, e_layers=2, d_model=128, d_ff=128, dropout=0.1):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.enc_in  = enc_in
        self.c_out   = c_out
        self.d_model = d_model
        self.dropout = dropout
        self.layer   = e_layers
        self.d_ff    = d_ff
        self.model = nn.ModuleList([TimesBlock(self.seq_len, 2, self.d_model, self.d_ff) for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self, x_enc, mask):

        means = torch.sum(x_enc, dim=1) / (torch.sum(mask == 1, dim=1) + 1e-5)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / (torch.sum(mask == 1, dim=1) + 1e-5))
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev + 1e-5

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)

        return dec_out

