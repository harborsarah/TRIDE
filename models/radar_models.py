import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class encoder_radar_2d3d(nn.Module):
    def __init__(self, params):
        super(encoder_radar_2d3d, self).__init__()
        self.params = params
        self.encoder2d = encoder_radar(params)
        self.encoder3d = RadarPointNet(params)
        self.feat_out_channels = self.encoder2d.feat_out_channels
    
    def forward(self, radar_channels, radar_points):
        skip_feat = self.encoder2d(radar_channels)
        point_feat = self.encoder3d(radar_points)

        return skip_feat, point_feat


class encoder_radar(nn.Module):
    def __init__(self, params):
        super(encoder_radar, self).__init__()

        self.params = params
        import torchvision.models as models
        self.conv = torch.nn.Sequential(nn.Conv2d(params.radar_input_channels, 3, 3, 1, 1, bias=False),
                                        nn.ELU())
        if params.encoder_radar == 'resnet34':
            self.base_model_radar = models.resnet34(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        elif params.encoder_radar == 'resnet18':
            self.base_model_radar = models.resnet18(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        else:
            print('Not supported encoder: {}'.format(params.encoder))
    def forward(self, x):
        feature = x
        feature = self.conv(feature)
        skip_feat = []
        i = 1
        for k, v in self.base_model_radar._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
            i = i + 1
        return skip_feat
    
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 160, 1)
        self.conv4 = torch.nn.Conv1d(160, 256, 1)
        self.fc1 = nn.Linear(256, 160)
        self.fc2 = nn.Linear(160, 64)
        self.fc3 = nn.Linear(64, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(160)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(160)
        self.bn6 = nn.BatchNorm1d(64)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out4 = F.relu(self.bn4(self.conv4(out3)))
        x = torch.max(out4, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x, [out1, out2, out3, out4]

class PointNetfeat(nn.Module):
    def __init__(self, k, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k)
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k)

    def forward(self, x):
        n_pts = x.size()[2]
        trans, feat = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat, feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat, feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)

            return torch.cat([x, pointfeat], 1), trans, trans_feat, feat

class RadarPointNet(nn.Module):
    def __init__(self, params, feature_transform=False):
        super(RadarPointNet, self).__init__()
        # in_channels = 
        self.feature_transform=feature_transform
        self.feat = PointNetfeat((params.radar_input_channels+2), global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, params.point_hidden_dim, 1)
        # self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(params.point_hidden_dim)
        # self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat, feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(batchsize, n_pts, -1)
        return x
    