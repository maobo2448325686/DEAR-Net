import torch
import torch.nn as nn
from decoder import decoder_grm
from pvtv2 import pvt_v2_b2
from models import edgeMod


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DEAMBlock(nn.Module):
    def __init__(self, in_planes, dilation):
        super(DEAMBlock, self).__init__()

        self.dilation = dilation

        self.edge = edgeMod.EGRAB(
            nn.Conv2d(in_planes, in_planes, 3, stride=1, padding=self.dilation, bias=False, dilation=self.dilation),
            in_planes, 3)

        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
        )
        self.cbam = CBAM(in_planes)

    def forward(self, x):
        x1 = self.conv(self.edge(x) + x)
        x2 = self.cbam(x1)
        return x2


class DEARNet(nn.Module):
    def __init__(self, backbone='pvt', output_stride=16, f_c=64, freeze_bn=False, in_c=3):
        super(DEARNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        # pvt
        self.backbone_pvt = pvt_v2_b2()  # [64, 128, 256, 512]
        path = './data/pretrained/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone_pvt.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone_pvt.load_state_dict(model_dict)

        self.decoder = decoder_grm(f_c, BatchNorm)

        self.deam1 = DEAMBlock(64, 1)
        self.deam2 = DEAMBlock(128, 2)
        self.deam3 = DEAMBlock(256, 3)
        self.deam4 = DEAMBlock(512, 3)

        self.conv_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0),
        )

        self.last = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)

        self.up_feature = nn.Upsample(scale_factor=2, mode='bilinear')
        self.down_channel = nn.Sequential(
            nn.Conv2d(320, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, hr_img1, hr_img2):
        p1 = self.backbone_pvt(hr_img1)
        p2 = self.backbone_pvt(hr_img2)

        y_1 = self.decoder(self.depm4(self.up_feature(p1[3])),
                           self.depm1(self.up_feature(p1[0])),
                           self.depm2(self.up_feature(p1[1])),
                           self.depm3(self.down_channel(self.up_feature(p1[2]))))

        y_2 = self.decoder(self.depm4(self.up_feature(p2[3])),
                           self.depm1(self.up_feature(p2[0])),
                           self.depm2(self.up_feature(p2[1])),
                           self.depm3(self.down_channel(self.up_feature(p2[2]))))

        feature = self.conv_final(torch.cat([y_1, y_2], dim=1))
        output = torch.sigmoid(feature)

        return output, feature
