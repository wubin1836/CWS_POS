# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from mydata import SegData
from PosSeg import Seg, SegGRU, SegBiGRU

use_cuda = torch.cuda.is_available()
net = SegBiGRU()
# net = SegGRU()

import json

with open('label2idx.json', 'r') as f: #2075
    label = json.load(f)

re_label = {}
for k, v in label.items():
    re_label[v] = k

re_label[101] = "SSS"
print re_label

if use_cuda:
    net = net.cuda()

net.load_state_dict(torch.load("models_bi/model_200.pt"))

learning_rate = 0.00001
BATCH_SIZE = 1

criterion = nn.CrossEntropyLoss(ignore_index=101)

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
dataLoader = torch.utils.data.DataLoader(SegData(source_file="test/source.txt", target_file="test/target.txt"), batch_size=BATCH_SIZE, shuffle=False, num_workers= 1)

INPUT_LENGTH= 40

sources = []
with open("test/source.txt") as f:
    lines = f.readlines()
    for line in lines:
        if len(line.split()) <= 40:
            sources.append(line)

fw = open("result.txt", "w")

for batch_idx, data_tuple in enumerate(dataLoader):
    input, output = data_tuple[0], data_tuple[1]
    if use_cuda:
        input, output = input.cuda(), output.cuda()

    input, output = Variable(input), Variable(output)
    pre_out = net(input)

    output = output.view(len(data_tuple[0]) * INPUT_LENGTH)

    _, predicted = torch.max(pre_out, 1)

    predicted = predicted.data.cpu()
    output = output.data.cpu()
    tmp = sources[batch_idx].split()

    ls = ""
    lt = ""
    for i in range(len(tmp)):
        #print tmp[i], re_label[output[i]], re_label[predicted[i]]
        ls = ls + tmp[i] + ":" + re_label[output[i]] + " "
        lt = lt + tmp[i] + ":" + re_label[predicted[i]] + " "

    fw.write(ls.strip() + "\n")
    fw.write(lt.strip() + "\n\n")