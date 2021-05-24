import torch.nn as nn

class Lstm(nn.Module):

    def __init__(self, input_size=184, hidden_size=128, num_layers=1, output_size=184*3, dropout=0, batch_first=True):
        super(Lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape  2层厚；就1一层而言，行是每个样本，列是特征
        # out也是2层
        out = self.linear(hidden[0])
        return out
