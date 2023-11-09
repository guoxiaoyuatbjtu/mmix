import os
import torch
import torch.optim as optim

import numpy as np

import utils
import argparse
import data_loader
import os

from models import brits

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--input_size', type=int, default=504)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_len', type=int, default=48)
parser.add_argument('--hid_size', type=int, default=64)
parser.add_argument('--aug', type=str, default='mmix')
parser.add_argument('--rate', type=float, default=0.1)
parser.add_argument('--k', type=int, default=48)
parser.add_argument('--reweight', action='store_true', default=False)
parser.add_argument('--data', type=str, default='NYCBike')
parser.add_argument('--pattern', type=str, default='PatternA')
args = parser.parse_args()

if args.aug == 'mmix':
    prefix = 'k_{}'.format(args.k)

if args.aug == 'unif':
    prefix = 'rate_{}'.format(args.rate)

if args.aug == 'none':
    prefix = 'none'

if args.aug == 'gaus':
    prefix = 'gaus'

if args.aug == 'temp':
    prefix = 'temp'

if args.aug == 'spat':
    prefix = 'spat'

if args.aug == 'spat_temp':
    prefix = 'spat_temp'

dirpath = './logs/{}/{}/{}/'.format(args.data, args.pattern, args.aug)

if not os.path.exists(dirpath):
    os.makedirs('./logs/{}/{}/{}/'.format(args.data, args.pattern, args.aug))

fout = open(dirpath + prefix + '.log', 'w')
seed = torch.Tensor(np.load(open('json/{}_{}.seed_{}'.format(args.data, args.pattern, args.k), 'rb'))).cuda()

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data_iter = data_loader.get_loader(datapath='json/{}_{}.json'.format(args.data, args.pattern), batch_size=args.batch_size)

    for epoch in range(args.epochs):

        model.train()
        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)
            run_loss += ret['loss'].item()

        evaluate(model, data_iter, epoch)


def evaluate(model, val_iter, epoch):
    model.eval()

    evals = []
    imputations = []
    # save_impute = []
    loss  = []

    for _, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        # save_impute.append(ret['imputations'])

        eval_masks = (1 - ret['masks']).data.cpu().numpy()
        eval_ = ret['values'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        loss.append(ret['loss'].cpu().item())

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    # save_impute = torch.cat(save_impute, dim=0)
    # np.save(open('logs/{}_{}_{}.data'.format(args.data, args.pattern, args.aug), 'wb'), save_impute.data.cpu().numpy())

    loss = sum(loss) / len(loss)
    mae  = np.abs(evals - imputations).mean()
    mse  = ((evals - imputations) ** 2).mean()
    rmse = np.sqrt(mse)
    fout.write("Epoch {:0>3d}\tLOSS: {:.2f}\tMAE: {:.2f}\tRMSE: {:.2f}\n".format(epoch, loss, mae, rmse))
    fout.flush()



def run():
    model = brits.Model(args.input_size, args.seq_len, args.hid_size, args.aug, {'rate': args.rate, 'k':args.k, 'seed': seed})

    if torch.cuda.is_available():
        model = model.cuda()
    train(model)
    fout.close()


if __name__ == '__main__':
    run()
