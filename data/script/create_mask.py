import numpy as np
  
name = 'SeattleCycle'
data = np.load(open('complete/{}.npy'.format(name), 'rb'))
mask = np.load(open('pattern/pattern_mt.npy', 'rb'))
mask = mask.repeat(2, axis=1).repeat(4, axis=0)
mask = mask[:data.shape[0], :data.shape[1]]
np.save('mask/{}_PatternB.npy'.format(name), mask)
