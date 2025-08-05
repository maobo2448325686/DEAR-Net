import torch
import torch.nn as nn
from decoder import decoder_grm
from models.pvtv1 import pvt_tiny
from models.resnet import build_backbone
from pvtv2 import pvt_v2_b2
from models import edgeMod
from transformers import AutoModel


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

        self.edge = edgeMod.ERAM(
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
    def __init__(self, backbone='pvtv2', output_stride=16, f_c=64):
        super(DEARNet, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.backbone = backbone

        if self.backbone == "pvtv2":
            # ============================ pvtv2 ============================
            self.backbone = pvt_v2_b2()  # [64, 128, 256, 512]
            path = './data/pretrained/pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        elif self.backbone == "resnet50":
            # ============================ ResNet50 ============================
            self.backbone = build_backbone(backbone="resnet50", output_stride=16)

            self.down_channel1_res50 = nn.Sequential(
                nn.Conv2d(256, 64, 1),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            self.down_channel2_res50 = nn.Sequential(
                nn.Conv2d(512, 128, 1),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            self.down_channel3_res50 = nn.Sequential(
                nn.Conv2d(1024, 256, 1),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            self.down_channel4_res50 = nn.Sequential(
                nn.Conv2d(2048, 512, 1),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )

        elif self.backbone == "pvtv1":
            # ============================ pvtv1 ============================
            self.backbone = pvt_tiny()  # [64, 128, 256, 512]
            path = 'data/pretrained/pvt_tiny.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        elif self.backbone == "mambavision":
            # ============================ MambaVision ============================
            model_path = r"data/pretrained/mambavision"
            self.backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

            self.up_feature_m1 = nn.Sequential(
                nn.Conv2d(80, 64, 1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            self.up_feature_m2 = nn.Sequential(
                nn.Conv2d(160, 128, 1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            self.up_feature_m3 = nn.Sequential(
                nn.Conv2d(320, 256, 1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
            self.up_feature_m4 = nn.Sequential(
                nn.Conv2d(640, 512, 1, bias=False),
                nn.Upsample(scale_factor=2, mode='bilinear'))

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
        if self.backbone == "mambavision":
            _, p1 = self.backbone(hr_img1)
            _, p2 = self.backbone(hr_img2)
        else:
            p1 = self.backbone(hr_img1)
            p2 = self.backbone(hr_img2)

        if self.backbone == "pvtv2" or self.backbone == "pvtv1":
            # ============================ pvtv2 or pvtv1============================
            y_1 = self.decoder(self.depm4(self.up_feature(p1[3])),
                               self.depm1(self.up_feature(p1[0])),
                               self.depm2(self.up_feature(p1[1])),
                               self.depm3(self.down_channel(self.up_feature(p1[2]))))

            y_2 = self.decoder(self.depm4(self.up_feature(p2[3])),
                               self.depm1(self.up_feature(p2[0])),
                               self.depm2(self.up_feature(p2[1])),
                               self.depm3(self.down_channel(self.up_feature(p2[2]))))

        elif self.backbone == "resnet50":
            # ============================ ResNet50 ============================
            y_1 = self.decoder(self.depm4(self.down_channel4_res50(p1[0])),
                               self.depm1(self.down_channel1_res50(p1[1])),
                               self.depm2(self.down_channel2_res50(p1[2])),
                               self.depm3(self.down_channel3_res50(p1[3])))

            y_2 = self.decoder(self.depm4(self.down_channel4_res50(p2[0])),
                               self.depm1(self.down_channel1_res50(p2[1])),
                               self.depm2(self.down_channel2_res50(p2[2])),
                               self.depm3(self.down_channel3_res50(p2[3])))

        elif self.backbone == "mambavision":
            # ============================ MambaVision ============================
            y_1 = self.decoder(self.depm4(self.up_feature_m4(p1[3])),
                               self.depm1(self.up_feature_m1(p1[0])),
                               self.depm2(self.up_feature_m2(p1[1])),
                               self.depm3(self.up_feature_m3(p1[2])))

            y_2 = self.decoder(self.depm4(self.up_feature_m4(p2[3])),
                               self.depm1(self.up_feature_m1(p2[0])),
                               self.depm2(self.up_feature_m2(p2[1])),
                               self.depm3(self.up_feature_m3(p2[2])))

        feature = self.conv_final(torch.cat([y_1, y_2], dim=1))
        output = torch.sigmoid(feature)

        return output
