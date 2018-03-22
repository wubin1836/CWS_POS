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
# net = SegBiGRU()
net = SegGRU()

if use_cuda:
    net = net.cuda()

# net.load_state_dict(torch.load("models/models_nas_darknet/12/7.pt"))

learning_rate = 0.00001
BATCH_SIZE = 512

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
dataLoader = torch.utils.data.DataLoader(SegData(source_file="train/source.txt", target_file="train/target.txt"), batch_size=BATCH_SIZE, shuffle=True, num_workers= 3)

dataValidLoader = torch.utils.data.DataLoader(SegData(source_file="dev/source.txt", target_file="dev/target.txt"), batch_size=BATCH_SIZE, shuffle=True, num_workers= 3)

def train():
    all_loss = 0
    all_step = 0
    for batch_idx, data_tuple in enumerate(dataLoader):
        input, output = data_tuple[0], data_tuple[1]
        if use_cuda:
            input, output = input.cuda(), output.cuda()

        input, output = Variable(input), Variable(output)
        pre_out = net(input)

        output = output.view(len(data_tuple[0]) * 40)
        # print output
        # print pre_out
        loss = criterion(pre_out, output)

        all_loss += loss.data[0]
        all_step += 1

        if all_step % 20 == 0:
            print all_loss / all_step

        loss.backward()
        optimizer.step()

    valid_loss = 0
    valid_step = 0
    for batch_idx, data_tuple in enumerate(dataValidLoader):
        input, output = data_tuple[0], data_tuple[1]
        if use_cuda:
            input, output = input.cuda(), output.cuda()

        input, output = Variable(input), Variable(output)
        pre_out = net(input)

        output = output.view(len(data_tuple[0]) * 40)

        loss = criterion(pre_out, output)
        valid_loss += loss.data[0]
        valid_step += 1

        _, predicted = torch.max(pre_out, 1)

        if batch_idx % 101 == 0:
            print predicted
            print output
            print "##############"
        loss.backward()
        optimizer.step()

    print "train: ", all_loss / all_step
    print "valid: ", valid_loss / valid_step



def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(1000):
    adjust_learning_rate(optimizer, epoch)
    print "epoch : "+ str(epoch)
    train()
    if epoch % 50 == 0:
        torch.save(net.state_dict(), "models/model_" + str(epoch) + ".pt")
    print "#####################################"
