import torch
import torch.nn as nn
from models import common
from models import dearnet


class CoreModule(nn.Module):
    # ERAM
    def __init__(self, features, M, r, L=32):

        super(CoreModule, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        self.edgeconvs = nn.ModuleList([])
        self.recon_trunk = common.ResidualBlock_noBN(nf=features, at='relu')
        self.conv3x3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        EC_combination = ['conv1-sobelx', 'conv1-sobelxy', 'conv1-sobely', 'conv1-sobelyx', 'conv1-laplacian']
        for i in range(len(EC_combination)):
            self.edgeconvs.append(nn.Sequential(
                common.EdgeConv(EC_combination[i], features, features),
            ))
        self.conv_reduce = nn.Conv2d(features * len(EC_combination), features, kernel_size=1, padding=0)
        self.sa = dearnet.SpatialAttention()
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        based_x = x
        out = self.recon_trunk(based_x)
        for i, edgeconv in enumerate(self.edgeconvs):
            fea = edgeconv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        feas = self.conv_reduce(feas)
        feas = self.sa(feas) * feas
        feas_f = torch.cat([out.unsqueeze_(dim=1), feas.unsqueeze_(dim=1)], dim=1)
        fea_f_U = torch.sum(feas_f, dim=1)
        fea_f_s = fea_f_U.mean(-1).mean(-1)
        fea_f_z = self.fc(fea_f_s)
        for i, fc in enumerate(self.fcs):
            vector_f = fc(fea_f_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors_f = vector_f
            else:
                attention_vectors_f = torch.cat([attention_vectors_f, vector_f], dim=1)
        attention_vectors_f = self.softmax(attention_vectors_f)
        attention_vectors_f = attention_vectors_f.unsqueeze(-1).unsqueeze(-1)
        fea_v_out = (feas_f * attention_vectors_f).sum(dim=1)
        return fea_v_out
class ERAM(nn.Module):
    def __init__(self, conv, n_feat, flag, bias=True, bn=False, act=nn.PReLU()):

        super(ERAM, self).__init__()
        self.n_feat = n_feat
        modules_body = []
        for i in range(2):
            modules_body.append(conv)
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CoreModule(n_feat, M=2, r=2, L=32))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        return res
