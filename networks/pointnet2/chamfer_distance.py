from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn


class ComputeLoss3d(nn.Module):
    def __init__(self):
        super(ComputeLoss3d, self).__init__()

        self.mse_func = nn.MSELoss()
        self.cd_loss_fun = ComputeCDLoss()
        self.loss = None
        self.consistent_loss = None
        self.cd_loss = None

    def forward(self, gt_points, structure_points, transed_gt_points=None, transed_structure_points=None, trans_func_list=None):

        gt_points = gt_points.cuda()
        structure_points = structure_points.cuda()

        batch_size = gt_points.shape[0]
        pts_num = gt_points.shape[1]
        dim = 3
        stpts_num = structure_points.shape[1]

        self.cd_loss = self.cd_loss_fun(structure_points, gt_points)

        trans_num = 0
        if transed_structure_points is not None:
            transed_structure_points = transed_structure_points.cuda()
            transed_gt_points = transed_gt_points.cuda()
            trans_num = transed_structure_points.shape[0]
            self.cd_loss = self.cd_loss + self.cd_loss_fun(transed_structure_points.view(trans_num * batch_size, stpts_num, dim),
                                                                             transed_gt_points.view(trans_num * batch_size, pts_num, dim))
            self.consistent_loss = None
            for i in range(0, trans_num):
                tmp_structure_points = trans_func_list[i](structure_points)
                tmp_structure_points = tmp_structure_points.detach()
                tmp_structure_points.requires_grad = False
                tmp_consistent_loss = self.mse_func(tmp_structure_points, transed_structure_points[i])
                if self.consistent_loss is None:
                    self.consistent_loss = tmp_consistent_loss
                else:
                    self.consistent_loss = self.consistent_loss + tmp_consistent_loss
            self.consistent_loss = self.consistent_loss / trans_num * 1000


        self.cd_loss = self.cd_loss / (trans_num + 1)

        self.loss = self.cd_loss

        if transed_structure_points is not None:
            self.loss = self.loss + self.consistent_loss
        return self.loss

    def get_cd_loss(self):
        return self.cd_loss

    def get_consistent_loss(self):
        return self.consistent_loss



def query_KNN_tensor(points, query_pts, k):
    '''
       :param points: bn x n x 3
       :param query_pts: bn x m x 3
       :param k: num of neighbors
       :return: nb x m x k  ids, sorted_squared_dis
       '''

    diff = query_pts[:, :, None, :] - points[:, None, :, :]

    squared_dis = torch.sum(diff*diff, dim=3)  # bn x m x n
    sorted_squared_dis, sorted_idxs = torch.sort(squared_dis, dim=2)
    sorted_idxs = sorted_idxs[:, :, :k]
    sorted_squared_dis = sorted_squared_dis[:, :, :k]

    return sorted_idxs, sorted_squared_dis

def compute_chamfer_distance(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''




    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff*diff,  dim=3)
    dist1 = dist
    dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist1, dim=2)
    dist_min2, _ = torch.min(dist2, dim=2)

    return dist_min1, dist_min2

class ComputeCDLoss(nn.Module):
    def __init__(self):
        super(ComputeCDLoss, self).__init__()

    def forward(self, recon_points, gt_points):

        dist1, dist2 = compute_chamfer_distance(recon_points, gt_points)

        #loss = (torch.mean(dist1) + torch.mean(dist2)) / (recon_points.shape[0])
        loss = (torch.mean(dist1) + torch.mean(dist2)) / 2
        return loss