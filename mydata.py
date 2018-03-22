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

class SegData(data.Dataset):
    def __init__(self, source_file, target_file):
        sources = []
        targets = []

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
                t = [label2idx[item.decode("utf8")] for item in tmp]
                for i in range(len(t), INPUT_LENGTH):
                    t.append(len(label2idx))
                targets.append(t)

        self.sources = sources
        self.targets = targets

    def __getitem__(self, index):
        source, target = self.sources[index], self.targets[index]

        return np.array(source), np.array(target)

    def __len__(self):
        return len(self.sources)