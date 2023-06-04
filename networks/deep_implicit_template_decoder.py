#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from networks.simple_nvp_lstm_5 import SimpleNVP
from networks.pointnet2.pointnet2 import Pointnet2StructurePointNet

class SdfDecoder_pointnet(nn.Module):
    def __init__(self, c_dim=1, dim=3, hidden_dim=256, with_dropout=True):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, hidden_dim)
        self.fc_0 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.th = nn.Tanh()
        self.with_dropout = with_dropout
        if self.with_dropout:
            self.dropout = nn.Dropout(p=0.05)

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.actvn(self.fc_pos(p))
        if self.with_dropout:
            net = self.dropout(net)

        net = self.actvn(self.fc_0(net))
        if self.with_dropout:
            net = self.dropout(net)

        net = self.actvn(self.fc_1(net))
        if self.with_dropout:
            net = self.dropout(net)

        net = self.actvn(self.fc_2(net))
        if self.with_dropout:
            net = self.dropout(net)

        net = self.actvn(self.fc_3(net))
        if self.with_dropout:
            net = self.dropout(net)

        c = self.fc_c(net)
        c = self.th(c)

        return c


class Decoder(nn.Module):
    def __init__(self, latent_size, warper_kargs, decoder_kargs, sdf_mode='pointnet_nvp'):
        super(Decoder, self).__init__()
        layers = warper_kargs['steps']
        hidden_size = warper_kargs['hidden_size']
        self.sdf_mode = sdf_mode
        self.use_label = warper_kargs['use_label']
        self.use_sp_features = warper_kargs['use_sp_features']
        num_structure_points = warper_kargs['num_structural_points']
        self.feature_size = 64
        if self.use_label:
            warper_latent_size = latent_size + num_structure_points
        elif self.use_sp_features:
            warper_latent_size = latent_size + self.feature_size
        else:
            warper_latent_size = latent_size
        self.sp_decoder = Pointnet2StructurePointNet(num_structure_points=num_structure_points,
                                                     feature_size=self.feature_size)
        self.warper = SimpleNVP(layers, warper_latent_size, hidden_size, proj=True)

        self.sdf_decoder = SdfDecoder_pointnet()

    def forward(self, input, on_data, code_input=None, output_warped_points=False, output_warping_param=False,
                step=1.0, predict_sp=False, use_input_sp=False, mode='inverse'):
        B, N, _ = input.size()
        _, O, _ = on_data.size()
        if predict_sp:
            input_all = torch.cat([input, on_data], dim=1)
            input_sp, one_hot_points_sp_idx, sp_features, local_sp_features = self.sp_decoder(
                input_all, on_data, return_weighted_feature=True)

            if self.use_label:
                local_code = one_hot_points_sp_idx
                _, _, S = one_hot_points_sp_idx.size()

                group_idx = torch.arange(S, dtype=torch.long).cuda().view(1, 1, S).repeat([B, S, 1])
                group_idx = group_idx.transpose(1, 2)
                local_sp_code = torch.zeros(B, S, S).cuda().scatter_(-1, group_idx, 1.)
            elif self.use_sp_features:
                local_code = local_sp_features
                local_sp_code = sp_features  # sp points
            else:
                local_code = None
                local_sp_code = None
        else:
            local_code = None
            local_sp_code = None
        input_all = torch.cat([input, on_data], dim=1)
        warped_xyzs, _ = self.warper(input_all, code_input, mode, local_code)
        p_final = warped_xyzs[-1]
        if predict_sp:
            if use_input_sp:
                input_sp_detach = input_sp.detach().clone()
                warped_sp_, _ = self.warper(input_sp_detach, code_input, mode, local_sp_code)
                warped_sp = warped_sp_[-1]
            else:
                warped_sp, _, _, _ = self.sp_decoder(
                    p_final, p_final[:, N:, :], return_weighted_feature=True)

        if not self.training:
            x = self.sdf_decoder(p_final)
            x = x.reshape(B, -1, 1)
            if output_warped_points:
                p_final = p_final.reshape(B, -1, 3)
                if predict_sp:
                    return p_final, x, input_sp, warped_sp
                else:
                    return p_final, x
            else:
                return x
        else:   # training mode, output intermediate positions and their corresponding sdf prediction
            xs = []
            for p in warped_xyzs:
                xs_ = self.sdf_decoder(p)
                xs_ = xs_.reshape(B, -1, 1)
                xs.append(xs_)
            if output_warped_points:
                if predict_sp:
                    return warped_xyzs, xs, input_sp, warped_sp
                else:
                    return warped_xyzs, xs
            else:
                return xs

    def forward_template(self, input):
        input = input.unsqueeze(0)
        return self.sdf_decoder(input)

    def initialize_warper(self, input):
        output = self.warper(input)
        return output



