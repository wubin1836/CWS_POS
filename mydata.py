import torch.utils.data as data
import torchvision
import random
import numpy as np
import os
import pandas as pd
import torch
INPUT_LENGTH = 40
import json
with open('word2idx.json', 'r') as f:
    word2idx = json.load(f)

with open('label2idx.json', 'r') as f:
    label2idx = json.load(f)

BMSE = {
    "B": 0,
    "M": 1,
    "S": 2,
    "E": 3
}

class SegData(data.Dataset):
    def __init__(self, source_file, target_file):
        sources = []
        targets = []
        targets_s = []
        fh = open(source_file)
        index = 0
        for line in fh.readlines():
            tmp = line.split()
            if len(tmp) <= INPUT_LENGTH:
                s = [word2idx[item.decode("utf8")] for item in tmp]
                for i in range(len(s), INPUT_LENGTH):
                    s.append(len(word2idx))
                sources.append(s)

        fh = open(target_file)
        index = 0
        for line in fh.readlines():
            tmp = line.split()
            if len(tmp) <= INPUT_LENGTH:
                t = [label2idx[item] for item in tmp]
                ts = [BMSE[item.split("_")[0]] for item in tmp]
                for i in range(len(t), INPUT_LENGTH):
                    t.append(len(label2idx))
                    ts.append(4)
                targets.append(t)
                targets_s.append(ts)
        self.sources = sources
        self.targets = targets
        self.targets_s = targets_s

    def __getitem__(self, index):
        source, target, target_s = self.sources[index], self.targets[index], self.targets_s[index]

        return np.array(source), np.array(target), np.array(target_s)

    def __len__(self):
        return len(self.sources)