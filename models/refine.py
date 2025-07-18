import torch
import torch.nn as nn


class RefineMod(nn.Module):
    def __init__(self, in_planes):
        '''
            细化模块
        '''
        super(RefineMod, self).__init__()
        dilations = [3, 3, 3, 3]

        self.seq_first = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )

        self.seq = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )

        self.seq1d = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, 1, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )
        self.seq2d = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, 1, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )
        self.seq3d = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, 1, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )
        self.seq4d = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, 1, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )

    def forward(self, input):
        x = self.seq_first(input)

        x1d = self.seq1d(x)
        x2d = self.seq2d(x1d)
        x3d = self.seq3d(x2d)
        x4d = self.seq4d(x3d)
        xd = x1d + x2d + x3d + x4d

        x1 = self.seq(x)
        x2 = self.seq(x1)
        x3 = self.seq(x2)
        x4 = self.seq(x3)
        x = x1 + x2 + x3 + x4
        output = self.conv_last(x + xd) * input

        return torch.sigmoid(output)
