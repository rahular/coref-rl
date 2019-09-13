from scipy.stats import *
import numpy as np
import json
import sys

def get_avg(data):
    new_data = []
    for l1, l2, l3 in zip(data['muc'], data['b_cubed'], data['ceafe']):
        new_data.append([(l1[0]+l2[0]+l3[0])/3, (l1[1]+l2[1]+l3[1])/3, (l1[2]+l2[2]+l3[2])/3])
    return new_data

def get_stats(metric):
    print(metric)
    if metric == 'all':
        B = [sum(l)/3 for l in get_avg(BL)]
        S = [sum(l)/3 for l in get_avg(SYS)]
    else:        
        B = [sum(l)/3 for l in BL[metric]]
        S = [sum(l)/3 for l in SYS[metric]]
    B = np.array(B)
    S = np.array(S)
    print('Avg: {:.4f}, {:.4f}'.format(B.mean(), S.mean()))

    stat, p = ttest_rel(B, S)
    print('T-test -> stat: {:.2f}, p-val: {:.2f}'.format(stat, p))
    stat, p = wilcoxon(B, S)
    print('Wilcoxon -> stat: {:.2f}, p-val: {:.2f}'.format(stat, p))

    wins = 0
    for _ in range(N):
        indeces = np.random.randint(0, B.shape[0], B.shape[0])
        if S[indeces].mean() > B[indeces].mean():
            wins += 1
    print("Bootstrap: {:.2f}".format(1-(wins/N)))
    print('=' * 100)
    return 1-(wins/N)

if __name__ == '__main__':
    N = 10000
    BL = json.loads(open(sys.argv[1]).read())
    SYS = json.loads(open(sys.argv[2]).read())

    [get_stats(m) for m in ['muc', 'b_cubed', 'ceafe']]
    # get_stats('all')