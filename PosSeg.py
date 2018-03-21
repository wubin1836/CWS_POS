import torch
import torch.nn as nn
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
import torch.nn.functional as F

class SegBiGRU(nn.Module):
    def __init__(self):
        super(SegBiGRU, self).__init__()
        self.hidden_size = 128
        self.embedding_c = nn.Embedding(4450, self.hidden_size)
        self.embedding_a = nn.Embedding(4450, self.hidden_size)

        self.gru = nn.GRU(128, 128)

        self.softmax = torch.nn.Softmax(dim=1)
        self.classifer = nn.Linear(256, 102)

    def forward(self, input):
        batch_size = input.size()[0]
        embed_c = self.embedding_c(input).transpose(1, 0)
        embed_a = self.embedding_a(input).transpose(1, 0)

        input_length = 40

        encoder_hidden_a = Variable(torch.zeros(1, batch_size, self.hidden_size))
        encoder_hidden_c = Variable(torch.zeros(1, batch_size, self.hidden_size))
        encoder_hiddens_a = Variable(torch.zeros(40, batch_size, self.hidden_size))
        encoder_hiddens_c = Variable(torch.zeros(40, batch_size, self.hidden_size))

        encoder_hidden_a_r = Variable(torch.zeros(1, batch_size, self.hidden_size))
        encoder_hidden_c_r = Variable(torch.zeros(1, batch_size, self.hidden_size))
        encoder_hiddens_a_r = Variable(torch.zeros(40, batch_size, self.hidden_size))
        encoder_hiddens_c_r = Variable(torch.zeros(40, batch_size, self.hidden_size))

        if use_cuda:
            encoder_hidden_a = encoder_hidden_a.cuda()
            encoder_hidden_c = encoder_hidden_c.cuda()
            encoder_hiddens_a = encoder_hiddens_a.cuda()
            encoder_hiddens_c = encoder_hiddens_c.cuda()

            encoder_hidden_a_r = encoder_hidden_a_r.cuda()
            encoder_hidden_c_r = encoder_hidden_c_r.cuda()
            encoder_hiddens_a_r = encoder_hiddens_a_r.cuda()
            encoder_hiddens_c_r = encoder_hiddens_c_r.cuda()


        for ei in range(input_length):
            #forward gru
            encoder_output, encoder_hidden_a = self.gru(
                embed_a[ei].contiguous().view(1, batch_size, -1), encoder_hidden_a)
            encoder_hiddens_a[ei] = encoder_hidden_a

            encoder_output, encoder_hidden_c = self.gru(
                embed_c[ei].contiguous().view(1, batch_size, -1), encoder_hidden_c)
            encoder_hiddens_c[ei] = encoder_hidden_c

            #backward gru
            encoder_output, encoder_hidden_a_r = self.gru(
                embed_a[input_length - 1 - ei].contiguous().view(1, batch_size, -1), encoder_hidden_a_r)
            encoder_hiddens_a[input_length - 1 - ei] = encoder_hidden_a_r

            encoder_output, encoder_hidden_c_r = self.gru(
                embed_c[input_length - 1 - ei].contiguous().view(1, batch_size, -1), encoder_hidden_c_r)
            encoder_hiddens_c[input_length - 1 - ei] = encoder_hidden_c_r

        hidden_a = torch.cat((encoder_hiddens_a, encoder_hiddens_a_r), dim=2)
        hidden_c = torch.cat((encoder_hiddens_c, encoder_hiddens_c_r), dim=2)

        hidden_a = hidden_a.transpose(1, 0)
        hidden_c = hidden_c.transpose(1, 0)

        matrix = torch.bmm(hidden_a, hidden_a.transpose(2, 1))
        attention = self.softmax(matrix).transpose(2, 1)

        hidden = torch.bmm(hidden_c.transpose(2,1), attention).transpose(1, 2)
        hidden = hidden.contiguous().view(-1, 256)
        hidden = self.classifer(hidden)
        hidden = self.softmax(hidden)

        return hidden


class SegGRU(nn.Module):
    def __init__(self):
        super(SegGRU, self).__init__()
        self.hidden_size = 128
        self.embedding_c = nn.Embedding(4450, self.hidden_size)
        self.embedding_a = nn.Embedding(4450, self.hidden_size)

        self.gru = nn.GRU(128, 128)

        self.softmax = torch.nn.Softmax(dim=1)
        self.classifer = nn.Linear(128, 102)

    def forward(self, input):
        batch_size = input.size()[0]

        embed_c = self.embedding_c(input).transpose(1, 0)
        embed_a = self.embedding_a(input).transpose(1, 0)

        input_length = 40

        encoder_hidden_a = Variable(torch.zeros(1, batch_size, self.hidden_size))
        encoder_hidden_c = Variable(torch.zeros(1, batch_size, self.hidden_size))
        encoder_hiddens_a = Variable(torch.zeros(40, batch_size, self.hidden_size))
        encoder_hiddens_c = Variable(torch.zeros(40, batch_size, self.hidden_size))

        if use_cuda:
            encoder_hidden_a = encoder_hidden_a.cuda()
            encoder_hidden_c = encoder_hidden_c.cuda()
            encoder_hiddens_a = encoder_hiddens_a.cuda()
            encoder_hiddens_c = encoder_hiddens_c.cuda()

        for ei in range(input_length):
            encoder_output, encoder_hidden_a = self.gru(
                embed_a[ei].contiguous().view(1, batch_size, -1), encoder_hidden_a)
            encoder_hiddens_a[ei] = encoder_hidden_a

        for ei in range(input_length):
            encoder_output, encoder_hidden_c = self.gru(
                embed_c[ei].contiguous().view(1, batch_size, -1), encoder_hidden_c)
            encoder_hiddens_c[ei] = encoder_hidden_c


        hidden_a = encoder_hiddens_a.transpose(1, 0)
        hidden_c = encoder_hiddens_c.transpose(1, 0)

        matrix = torch.bmm(hidden_a, hidden_a.transpose(2, 1))
        attention = self.softmax(matrix).transpose(2, 1)

        hidden = torch.bmm(hidden_c.transpose(2,1), attention).transpose(1, 2)
        hidden = hidden.contiguous().view(-1, 128)
        hidden = self.classifer(hidden)
        hidden = self.softmax(hidden)

        return hidden



class Seg(nn.Module):
    def __init__(self):
        super(Seg, self).__init__()
        self.embedding_c = nn.Embedding(4450, 128)
        self.embedding_a = nn.Embedding(4450, 128)
        self.softmax = torch.nn.Softmax(dim=1)
        self.classifer = nn.Linear(128, 102)

    def forward(self, input):

        embed_c = self.embedding_c(input)
        embed_a = self.embedding_a(input)

        matrix = torch.bmm(embed_a, embed_a.transpose(2, 1))

        attention = self.softmax(matrix).transpose(2, 1)

        hidden = torch.bmm(embed_c.transpose(2,1), attention).transpose(1, 2)


        hidden = hidden.contiguous().view(-1, 128)
        hidden = self.classifer(hidden)
        hidden = self.softmax(hidden)


        return hidden

