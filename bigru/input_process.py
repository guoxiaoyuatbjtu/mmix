# coding: utf-8

import numpy as np
import json

def parse_delta(masks, input_size):
    deltas = []
    for time_step in range(48):
        if time_step == 0:
            deltas.append(np.ones(input_size))
        else:
            deltas.append(np.ones(input_size) + (1 - masks[time_step]) * deltas[-1])
    return np.asarray(deltas)

# Real missing, missing for test
data = np.load(open('../data/complete/SeattleCycle.npy', 'rb'))
masks = np.load(open('../data/mask/SeattleCycle_PatternC.npy', 'rb'))

data_len = data.shape[0]
data = data.reshape(data_len, -1)
data_len = data_len - data_len % 48

data = data[:data_len]
masks = masks[:data_len]

data = data.reshape(int(data_len / 48), 48, -1)
masks = masks.reshape(int(data_len / 48), 48, -1)

fs = open('json/SeattleCycle_PatternC.json', 'w')

for i in range(int(data_len / 48)):
    sample = {}
    sample['forward'] = {}
    f_values = data[i]
    f_masks  = masks[i]
    f_deltas = parse_delta(f_masks, input_size=data.shape[-1])
    sample['forward']['masks']  = f_masks.tolist()
    sample['forward']['values'] = f_values.tolist()
    sample['forward']['deltas'] = f_deltas.tolist()

    sample['backward'] = {}
    b_values = data[i][::-1]
    b_masks  = masks[i][::-1]
    b_deltas = parse_delta(b_masks, input_size=data.shape[-1])
    sample['backward']['masks']  = b_masks.tolist()
    sample['backward']['values'] = b_values.tolist()
    sample['backward']['deltas'] = b_deltas.tolist()

    fs.write(json.dumps(sample) + '\n')


fs.close()

