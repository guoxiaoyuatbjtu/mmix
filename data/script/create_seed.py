import numpy as np
  
def create_seed(seed, k):
    chunks = []
    i = k
    p = 0
    while i < seed.shape[0] - k:
        if seed[i - k: i + 2 * k].sum() == 0:
            chunk = seed[p:i]
            if chunk.sum() != 0:
                chunks.append(chunk)
            p = i + k
            i = i + 2 * k
        else:
            i = i + 1
    chunks.append(seed[p:])
    seed = np.vstack(chunks)
    return seed

name = 'SeattleCycle_PatternB'

for k in [6, 12, 24, 48]:
    data = np.load(open('mask/{}.npy'.format(name), 'rb'))
    seed = 1 - data
    seed = create_seed(seed, k)
    seed = 1 - seed
    np.save(open('mask/{}.seed_{}'.format(name, k), 'wb'), seed)

