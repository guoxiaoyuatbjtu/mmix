import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='Times New Roman', size=16)

# dataset = 'PEMS-BAY'
dataset = 'SeattleCycle'
# pattern = 'PatternA' # METR-LA
pattern = 'PatternB' # PM2.5

models = ['bilstm', 'bigru', 'bitcn', 'brits']
augs   = ['none', 'unif', 'temp', 'spat', 'spat_temp', 'mmix']

augs_fn = {
    'none': 'none.log',
    'unif': 'rate_0.1.log',
    'temp': 'temp.log',
    'spat': 'spat.log',
    'spat_temp': 'spat_temp.log',
    'mmix': 'k_6.log'
}

augs_lb = {
    'none': 'Vanilla',
    'unif': 'Point',
    'temp': 'Temporal',
    'spat': 'Spatial',
    'spat_temp': 'Spatiotemporal',
    'mmix': 'M-Mix'
}


for name in models:
    plt.clf()
    plt.figure(figsize=(10, 6))    
    for aug in augs:
        log_fin = open('{}/logs/{}/{}/{}/{}'.format(name, dataset, pattern, aug, augs_fn[aug]))
        lines = log_fin.readlines()
        lines = list(map(lambda x: x.strip().split(), lines))
        mae   = list(map(lambda x: float(x[-3]), lines))
        x = [ i for i in range(500)]
        plt.plot(x, mae, label='{}'.format(augs_lb[aug]))
        print('{}_{}_{:.2f}'.format(name, aug, sum(mae[-10:])/10))
    plt.legend(loc='upper center', ncol=3)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.savefig('mae_{}_{}_{}.pdf'.format(name, dataset, pattern), format='pdf')

for name in models:
    plt.clf()
    plt.figure(figsize=(10, 6))    
    for aug in augs:
        log_fin = open('{}/logs/{}/{}/{}/{}'.format(name, dataset, pattern, aug, augs_fn[aug]))
        lines = log_fin.readlines()
        lines = list(map(lambda x: x.strip().split(), lines))
        rmse  = list(map(lambda x: float(x[-1]), lines))
        x = [ i for i in range(500)]
        plt.plot(x, rmse, label='{}'.format(augs_lb[aug]))
        print('{}_{}_{:.2f}'.format(name, aug, sum(rmse[-10:])/10))
    plt.legend(loc='upper center', ncol=3)
    plt.xlabel('Epoch')
    plt.ylabel('Root Mean Square Error')
    plt.savefig('rmse_{}_{}_{}.pdf'.format(name, dataset, pattern), format='pdf')
