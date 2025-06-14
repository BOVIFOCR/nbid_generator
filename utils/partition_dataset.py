from glob import glob
import re
import math
import random
import sys
import argparse
import os

import numpy as np

def expand_line(line, id):
    els = line.split(',')
    processed = [str(id)]
    nb = ','.join(els[:-2])

    #if els[-2] == "" or els[-2] == " ":
    #    return None
    if 'assin' in els[-1]:
        return None

    if "[" not in nb and "]" not in nb:
        coors = (int(els[0]), int(els[1]))
        size = (int(els[2]), int(els[3]))

        processed.append(str(coors[0]))
        processed.append(str(coors[1]))

        processed.append(str(coors[0] + size[0]))
        processed.append(str(coors[1]))

        processed.append(str(coors[0] + size[0]))
        processed.append(str(coors[1] + size[1]))

        processed.append(str(coors[0]))
        processed.append(str(coors[1] + size[1]))

        processed = processed + els[4:]

    else:
        processed.append(re.sub("[\[\]]", "", els[0][1:]))
        processed.append(re.sub("[\[\]]", "", els[4][1:])) # Extra space at the start.

        processed.append(re.sub("[\[\]]", "", els[3][1:]))
        processed.append(re.sub("[\[\]]", "", els[7][1:]))

        processed.append(re.sub("[\[\]]", "", els[2][1:]))
        processed.append(re.sub("[\[\]]", "", els[6][1:]))

        processed.append(re.sub("[\[\]]", "", els[1][1:]))
        processed.append(re.sub("[\[\]]", "", els[5][1:]))

        processed = processed + [els[-2], els[-1]] # Skipping -1 -1

    processed = ','.join(processed)
    return processed

def gen_new_file(name):
    file_name = name
    file = open(file_name, "r", encoding='unicode_escape')

    new_file_name = file_name.replace("_GT.txt", ".tsv")
    new_file = open(new_file_name, "w")

    in_lines = file.readlines()
    id = 0
    out_lines = []
    for line in in_lines[1:]:
        out_line = expand_line(line.replace("\n", ""), id)
        if out_line is not None:
            out_line = out_line.split(',')
            out_line[-1] = out_line[-1][1:]
            out_line[-2] = out_line[-2][1:]
            out_lines.append(",".join(out_line))
            id += 1
    for i in range(0, len(out_lines)):
        new_file.write(out_lines[i] + "\n")

    new_file.close()
    file.close()

# Legacy for retrocompatibility with old BID.
def gen_new_annotations(fdir):
    filenames = glob(f"{fdir}/*_GT.txt")
    for f in filenames:
        gen_new_file(f)


def partition_dataset(mode, filenames, ratio_train=60, ratio_valid=20, ratio_test=20):
    if ratio_train + ratio_test + ratio_valid != 100:
        raise ValueError("Partition ratios do not add to 100%.")
    if mode not in ['cross', 'std']:
        raise ValueError("Mode must be either cross or std.")

    if mode == 'cross':
        fs = {}
        for filename in filenames:
            idx = filename.split('/')[-1].split('_')[0]
            if idx not in fs:
                fs[idx] = []
            fs[idx].append(filename)

        idxs = sorted(fs.keys())
        random.shuffle(idxs)
        train_idx, valid_idx, test_idx = np.array_split(
            np.array(idxs),
            np.array([int(ratio_train*len(idxs)/100), int((1 - ratio_valid)*len(idxs)/100)])
        )

        train = [x for y in train_idx for x in fs[y]]
        valid = [x for y in valid_idx for x in fs[y]]
        test = [x for y in test_idx for x in fs[y]]

    else:
        fs = filenames

        train = fs[:int(ratio_train*len(fs))]
        valid = fs[int(ratio_train*len(fs)):int(((1 - ratio_valid)/2)*len(fs))]
        test = fs[int(((1 - ratio_valid)/2)*len(fs)):]

    train = [x.replace(".tsv", ".jpg") for x in train]
    valid = [x.replace(".tsv", ".jpg") for x in valid]
    test = [x.replace(".tsv", ".jpg") for x in test]
    random.shuffle(train)
    random.shuffle(test)
    random.shuffle(valid)

    return train, valid, test

def make_prots(files):

    fs = set([x.split('.')[0] for x in files])
    fs = list(fs)

    inst50_fs = fs[:math.ceil(len(fs)*0.6)]
    inst50_full = [x for x in files if x.split('.')[0] in inst50_fs]

    inst25_fs = fs[:math.ceil(len(fs)*0.4)]
    inst25_full = [x for x in files if x.split('.')[0] in inst25_fs]

    dup50_full = []
    dup25_full = []

    dct = {}
    for f in fs:
        dct[f] = [x for x in files if x.split('.')[0] == f]
        n_inst = len(dct[f])

        dup50_full += dct[f][:math.ceil(n_inst/2)]
        dup25_full += dct[f][:math.ceil(n_inst/4)]

    random.shuffle(dup50_full)
    random.shuffle(dup25_full)

    return {
        'inst50': inst50_full,
        'inst25': inst25_full,
        'dup50': dup50_full,
        'dup25': dup25_full
    }



def save_dataset(train, valid, test, other=None, fdir=".", partition=""):
    if partition != "":
        preamb = partition.strip('/') + "_"
    else:
        preamb = ""
    with open(f"{fdir+partition}{preamb}train.csv", "w") as fd:
        for idx, f in enumerate(train):
            fd.write(f"{idx},RG,{f.split('/')[-1]}\n")
    with open(f"{fdir+partition}{preamb}valid.csv", "w") as fd:
        for idx, f in enumerate(valid):
            fd.write(f"{idx},RG,{f.split('/')[-1]}\n")
    with open(f"{fdir+partition}{preamb}test.csv", "w") as fd:
        for idx, f in enumerate(test):
            fd.write(f"{idx},RG,{f.split('/')[-1]}\n")
    if other is not None:
        for prot in other:
            with open(f"{fdir+partition}{preamb}{prot}.csv", "w") as fd:
                for idx, f in enumerate(other[prot]):
                    fd.write(f"{idx},RG,{f.split('/')[-1]}\n")


def main(mode, fdir, partition, ratio_train=60, ratio_valid=20, ratio_test=20, gen_new=True):
    if mode not in ['cross', 'std']:
        print("Mode must be cross or std.")
        exit(-1)

    if ratio_train + ratio_test + ratio_valid != 100:
        print("Partition ratios must add up to 100%.")
        exit(-1)

    if gen_new == True:
        fs = glob(f"{fdir}labels/*.txt")
        gen_new_annotations(fs)
    else:
        fs = glob(f"{fdir+partition}labels/*.tsv")

    train, valid, test = partition_dataset(mode, fs, ratio_train, ratio_valid, ratio_test)
    other = make_prots(train)

    save_dataset(train, valid, test, other, fdir, partition)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='cross', choices=['cross', 'std'])
    
    parser.add_argument('--train', default=80)
    parser.add_argument("--valid", default=10)
    parser.add_argument("--test", default=10)

    parser.add_argument("--dataset", default='./nbid_real/', required=True)
    parser.add_argument("--partition", default="")

    parser.add_argument("--gen_new", default=False, action="store_true")
    args = vars(parser.parse_args())

    if not os.path.isdir(args['dataset']):
        print("Error: fdir is not a directory.")
        exit(-1)

    main(args['mode'], args['dataset'], 
        ratio_train=int(args['train']), ratio_valid=int(args['valid']), ratio_test=int(args['test']),
        gen_new = args['gen_new'], partition=args['partition'])
