import random
import math

f = './train.csv'

with open(f, 'r') as fd:
    ls = [x.split(',')[-1] for x in fd]

fs = set([x.split('.')[0] for x in ls])

fs = list(fs)

inst50_fs = fs[:math.ceil(len(fs)/2)]
inst50_full = [x for x in ls if x.split('.')[0] in inst50_fs]

inst25_fs = fs[:math.ceil(len(fs)/4)]
inst25_full = [x for x in ls if x.split('.')[0] in inst25_fs]

dup50_full = []
dup25_full = []

dct = {}
for f in fs:
    dct[f] = [x for x in ls if x.split('.')[0] == f]
    n_inst = len(dct[f])

    dup50_full += dct[f][:n_inst//2]
    dup25_full += dct[f][:n_inst//4]

random.shuffle(dup50_full)
random.shuffle(dup25_full)

def save(fname, lst):
    with open(fname, 'w') as fd:
        for i,x in enumerate(lst):
            fd.write(str(i) + ',RG,' + x)

save('./inst50.csv', inst50_full)
save('./inst25.csv', inst25_full)
save('./dup50.csv', dup50_full)
save('./dup25.csv', dup25_full)

print(len(dup50_full))
print(len(dup25_full))

