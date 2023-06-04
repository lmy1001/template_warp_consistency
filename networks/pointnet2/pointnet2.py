from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
#from networks.pointnet2.pointnet2_modules import PointnetSAModuleMSG
from networks.pointnet2.pointnet2_utils import PointNetSetAbstractionMsg, query_sp_point
import torch.nn.functional as F


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)), inplace=True)
        return new_points

class Pointnet2StructurePointNet(nn.Module):
    def __init__(self, num_structure_points, feature_size, input_channels=3):
        super(Pointnet2StructurePointNet, self).__init__()
        self.point_dim = 3
        self.num_structure_points = num_structure_points
        self.input_channels = input_channels
        self.SA_modules = nn.ModuleList()
        self.stpts_prob_map = None
        self.stpts_prob_feature_map = None

        '''
        self.SA_modules.append(
            PointNetSetAbstractionMsg(
                npoint=512,
                radius_list=[0.1, 0.2, 0.4],
                nsample_list=[16, 32, 128],
                in_channel=0,
                mlp_list=[
                    [32, 32, 64],
                    [64, 64, 128],
                    [64, 96, 128],
                ]
            )
        )

        input_channels = 64 + 128 + 128
        
        self.SA_modules.append(
            PointNetSetAbstractionMsg(
                npoint=128,
                radius_list=[0.2, 0.4, 0.8],
                nsample_list=[32, 64, 128],
                in_channel=0,
                mlp_list=[
                    [64, 64, 128],
                    [128, 128, 256],
                    [128, 128, 256],
                ]
            )
        )
        '''
        self.SA_modules.append(
            PointNetSetAbstractionMsg(
                npoint=128,
                radius_list=[0.1, 0.2, 0.4],
                nsample_list=[16, 32, 128],
                in_channel=0,
                mlp_list=[
                    [16, 16, 32],
                    [32, 64, 64],
                    [64, 96, 128],
                ]
            )
        )

        input_channels = 32 + 64 + 128

        self.SA_modules.append(
            PointNetSetAbstractionMsg(
                npoint=64,
                radius_list=[0.2, 0.4, 0.8],
                nsample_list=[16, 32, 128],
                in_channel=input_channels,
                mlp_list=[
                    [16, 16, 32],
                    [32, 64, 64],
                    [64, 96, 128],
                ]
            )
        )
        conv1d_stpts_feature_modules = []

        '''
        conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128 + 256 + 256, out_channels=512, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.BatchNorm1d(512))
        conv1d_stpts_prob_modules.append(nn.ReLU())
        in_channels = 512
        while in_channels >= self.num_structure_points * 2:
            out_channels = int(in_channels / 2)
            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(out_channels))
            conv1d_stpts_prob_modules.append(nn.ReLU())
            in_channels = out_channels

        conv1d_stpts_prob_modules.append(nn.Dropout(0.2))

        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=self.num_structure_points, kernel_size=1))
        '''
        conv1d_stpts_prob_modules = []
        conv1d_stpts_prob_modules_2 = []
        conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
        #conv1d_stpts_feature_modules.append(nn.Conv1d(in_channels=128 + 256 + 256, out_channels=512, kernel_size=1))
        #conv1d_stpts_feature_modules.append(nn.BatchNorm1d(512))
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels= 32 + 64 + 128, out_channels=128, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.BatchNorm1d(128))
        conv1d_stpts_prob_modules.append(nn.ReLU())
        #in_channels = 512
        in_channels = 128
        while in_channels >= self.num_structure_points * 2:
            out_channels = int(in_channels / 2)
            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(out_channels))
            conv1d_stpts_prob_modules.append(nn.ReLU())
            in_channels = out_channels

        conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
        # conv1d_stpts_prob_modules.append(
        #      nn.Conv1d(in_channels=in_channels, out_channels=feature_size, kernel_size=1))
        # conv1d_stpts_prob_modules.append(nn.BatchNorm1d(feature_size))
        # conv1d_stpts_prob_modules.append(nn.ReLU())
        # conv1d_stpts_prob_modules.append(
        #      nn.Conv1d(in_channels=feature_size, out_channels=self.num_structure_points, kernel_size=1))
        conv1d_stpts_prob_modules.append(
                nn.Conv1d(in_channels=in_channels, out_channels=self.num_structure_points, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.BatchNorm1d(self.num_structure_points))

        conv1d_stpts_prob_modules_2.append(nn.Softmax(dim=2))

        self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)
        self.conv1d_stpts_prob_2 = nn.Sequential(*conv1d_stpts_prob_modules_2)

        #self.conv1d_stpts_feature = nn.Sequential(*conv1d_stpts_feature_modules)

        conv1d_stpts_feature_modules.append(
                 nn.Conv1d(in_channels=64, out_channels=feature_size, kernel_size=1))
        conv1d_stpts_feature_modules.append(nn.BatchNorm1d(feature_size))
        self.conv1d_stpts_feature = nn.Sequential(*conv1d_stpts_feature_modules)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, input, on_data,  return_weighted_feature=False):
        '''
        :param pointcloud: input point cloud with shape (bn, num_of_pts, 3)
        :param return_weighted_feature: whether return features for the structure points or not
        :return:
        '''
        #pointcloud = pointcloud.cuda()
        pointcloud = on_data
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        #features = torch.transpose(features, 1, 2)
        temp_map = self.conv1d_stpts_prob(features)
        #print(temp_map.shape)
        self.stpts_prob_map = self.conv1d_stpts_prob_2(temp_map)
        #print(self.stpts_prob_map.shape)
        self.stpts_prob_feature_map = self.conv1d_stpts_feature(temp_map.transpose(2, 1).contiguous())
        #print(self.stpts_prob_feature_map.shape)
        weighted_xyz = torch.sum(self.stpts_prob_map[:, :, :, None] * xyz[:, None, :, :], dim=2)
        if return_weighted_feature:
            #weighted_features = torch.sum(self.stpts_prob_map[:, None, :, :] * self.conv1d_stpts_feature_map[:, :, None, :], dim=3)
            weighted_features = self.stpts_prob_feature_map  # B * feature_size * num_sp
            one_hot_points_sp_idx, local_features = query_sp_point(input, weighted_xyz, weighted_features)

        if return_weighted_feature:
            return weighted_xyz, one_hot_points_sp_idx, weighted_features.transpose(1,2).contiguous(),local_features
        else:
            return weighted_xyz

if __name__=='__main__':
    net = Pointnet2StructurePointNet(num_structure_points=16, input_channels=3)
    input = torch.randn(5,1000, 3)

    for epoch in range(20):
        net.train()
        weighted_xyz = net(input)

    print(weighted_xyz.shape)