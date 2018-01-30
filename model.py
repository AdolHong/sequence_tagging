import torch
import torch.nn as nn
from torch.autograd import Variable
from torchcrf import CRF

PAD = "<PAD>"
UNK = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


class BilstmCrf(nn.Module):
    def __init__(self, vocab_size, tag_to_idx, hidden_size, embed_size, num_layers, dropout):
        super(BilstmCrf, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.tag_to_idx = tag_to_idx
        self.tag_size = len(tag_to_idx)

        # architecture
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=self.hidden_size // 2,
            num_layers=self.num_layers,
            bias=True,
            dropout=dropout,
            bidirectional=True
        )
        # LSTM output to tag
        self.out = nn.Linear(self.hidden_size, self.tag_size)
        self.crflayer = CRF(self.tag_size)

        # use xavier_normal to init weight
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal(self.embed.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal(param)
        nn.init.xavier_normal(self.out.weight)

    def init_hidden(self, batch_size):  # initialize hidden states
        h = Variable(torch.zeros(self.num_layers * 2, batch_size,
                                 self.hidden_size // 2))  # hidden states
        c = Variable(torch.zeros(self.num_layers * 2, batch_size,
                                 self.hidden_size // 2))  # cell states
        return h, c

    def loss(self, x, y):
        mask = torch.autograd.Variable(x.data.gt(0))
        emissions = self.lstm_forward(x)
        return self.crflayer(emissions, y, mask=mask)

    def forward(self, x):
        """ prediction """
        mask = torch.autograd.Variable(x.data.gt(0))
        emissions = self.lstm_forward(x)
        return self.crflayer.decode(emissions, mask=mask)

    def lstm_forward(self, x):  # LSTM forward pass
        """
        lstm forward to calculate emissions(equals to probability of tags)
        """
        batch_size = x.size(1)
        # transpose:  (seq_len, batch_size) ==> (batch_size, seq_len)
        lengths = [self.len_unpadded(seq) for seq in x.transpose(0, 1)]
        embed = self.embed(x)  # batch, seq,tag

        # pack and unpack are excellent
        embed_pack = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths)
        lstm_output_pack, _ = self.lstm(
            embed_pack, self.init_hidden(batch_size=batch_size))
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_output_pack)

        emissions = self.out(lstm_output)
        return emissions

    def len_unpadded(self, x):
        """ get unpadded sequence length"""
        def scalar(x): return x.view(-1).data.tolist()[0]
        return next((i for i, j in enumerate(x) if scalar(j) == 0), len(x))
