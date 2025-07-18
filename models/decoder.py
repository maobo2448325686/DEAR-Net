import torch
import torch.nn as nn
import torch.nn.functional as F
from models import refine, dearnet


class GRM(nn.Module):
    def __init__(self, in_d, out_d):
        super(GRM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, 1, bias=False),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU()
        )
        self.ref = refine.RefineMod(self.out_d)
        self.spa = dearnet.SpatialAttention()
        self.cha = dearnet.ChannelAttention(self.in_d)

    def forward(self, input):
        x = self.conv1(input)

        x1 = self.conv1(self.spa(input)) * x
        x2 = self.conv1(self.cha(input)) * x
        x = self.ref(x)
        output = x + x1 + x2
        return output


class DecoderGRM(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(DecoderGRM, self).__init__()
        self.fc = fc
        self.grm1 = GRM(64, 64)
        self.grm2 = GRM(128, 64)
        self.grm3 = GRM(256, 64)
        self.grm4 = GRM(512, 64)
        self.last_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(128, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       )

        self._init_weight()

    def forward(self, feat1, feat2, feat3, feat4):
        # 以下四行是原模型代码
        x2 = self.grm1(feat2)
        x3 = self.grm2(feat3)
        x4 = self.grm3(feat4)
        x1 = self.grm4(feat1)

        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x2.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def decoder_grm(fc, BatchNorm):
    return DecoderGRM(fc, BatchNorm)
