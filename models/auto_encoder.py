import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, ch_in, drop_out=None):
        super(AutoEncoder, self).__init__()
        encoder_1 = nn.Linear(ch_in,  20, bias=False)
        encoder_2 = nn.Linear(20, 10, bias=False)
        if drop_out is not None:
            dp = nn.Dropout(drop_out)
        else:
            dp = nn.Identity()
        encoder_3 = nn.Linear(10, 5, bias=False)
        decoder_1 = nn.Linear(5, 10, bias=False)
        decoder_2 = nn.Linear(10, 20, bias=False)

        decoder_3 = nn.Linear(20, ch_in, bias=False)
        relu = nn.ReLU()
        self.net = nn.Sequential(encoder_1, relu, encoder_2, dp, relu, encoder_3, relu,
                                 decoder_1, relu, decoder_2, dp, relu, decoder_3)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    net = AutoEncoder(28)
    input = torch.zeros([1, 28])
    outout = net(input)
