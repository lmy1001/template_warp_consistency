import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn
import signal
import sys
import os
import logging
import math
import json
import time
import datetime
import random

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.lr_schedule import get_learning_rate_schedules
from generate_meshes_correspondence import save_to_ply
import plyfile
import deep_sdf.loss as loss
# from pytorch3d.loss.chamfer import chamfer_distance
from networks.pointnet2.pointnet2 import Pointnet2StructurePointNet
from networks.pointnet2.chamfer_distance import compute_chamfer_distance, ComputeLoss3d
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import matplotlib.pyplot as plt
import trimesh
from networks.pointnet2.pointnet2_utils import farthest_point_sample, index_points

def apply_sp_loss(loss_func, input_sp, input_xyz):
    #loss, _ = chamfer_distance(input_sp, input_xyz, point_reduction='mean')
    loss = loss_func(input_xyz, input_sp)
    return loss

def apply_sp_loss_reg(warped_sp_, input_sp, huber_fn, d=0.25):
    pw_loss = []
    t = 2
    for k in range(len(warped_sp_)):
        if len(warped_sp_) == 1 or k % t == t-1:
            dist = torch.norm(warped_sp_[k] - input_sp, dim=-1)
            pw_loss.append(huber_fn(dist, delta=d))
    pw_loss = sum(pw_loss) / len(pw_loss)
    return pw_loss

def apply_curriculum_l1_loss(pred_sdf_list, sdf_gt, loss_l1_soft, num_sdf_samples):

    soft_l1_eps_list = [2.5e-2, 2.5e-3, 0]
    soft_l1_lamb_list = [0, 0.2, 0.5]

    sdf_loss = []
    t = 2
    for k in range(len(pred_sdf_list)):
        if k % t == t-1:
            eps = soft_l1_eps_list[k // t]
            lamb = soft_l1_lamb_list[k // t]
            l = loss_l1_soft(pred_sdf_list[k][:, :num_sdf_samples, :1], sdf_gt[:, :num_sdf_samples, :1],
                             eps=eps, lamb=lamb)
            # l = loss_l1(pred_sdf_list[k], sdf_gt[i].cuda()) / num_sdf_samples
            sdf_loss.append(l)
    sdf_loss = sum(sdf_loss) / len(sdf_loss)
    return sdf_loss

def apply_pointwise_reg(warped_xyz_list, xyz_, huber_fn, num_sdf_samples, d=0.25):
    pw_loss = []
    t = 2
    for k in range(len(warped_xyz_list)):
        if len(warped_xyz_list) == 1 or k % t == t-1:
            dist = torch.norm(warped_xyz_list[k][:, :num_sdf_samples, :3] - xyz_[:, :num_sdf_samples, :3], dim=-1)
            pw_loss.append(huber_fn(dist, delta=d))
    pw_loss = sum(pw_loss) / len(pw_loss)
    return pw_loss

def apply_pointpair_reg(warped_xyz, xyz_, loss_lp, scene_per_batch):
    delta_xyz = warped_xyz - xyz_
    xyz_reshaped = xyz_.view((scene_per_batch, -1, 3))
    num_points = xyz_reshaped.shape[1]
    delta_xyz_reshape = delta_xyz.view((scene_per_batch, -1, 3))
    k = xyz_reshaped.shape[1] // 8
    lp_loss =  torch.sum(loss_lp(
        xyz_reshaped[:, :k].view(scene_per_batch, -1, 1, 3),
        xyz_reshaped[:, k:].view(scene_per_batch, 1, -1, 3),
        delta_xyz_reshape[:, :k].view(scene_per_batch, -1, 1, 3),
        delta_xyz_reshape[:, k:].view(scene_per_batch, 1, -1, 3),
    )) / (scene_per_batch * num_points)
    return lp_loss

def apply_vol_loss(loss_func, sp, input):
    loss = loss_func(sp, input)
    return loss

def save_to_ply(verts, verts_warped, ply_filename_out):
    num_verts = verts.shape[0]
    num_verts_2 = verts_warped.shape[0]
    verts_color_0 = [255, 0, 0]
    verts_color_1 = [0, 255, 255]

    verts_tuple = np.zeros(
        (num_verts+num_verts_2,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])

    for i in range(0, num_verts):
        verts_tuple[i] = (verts[i][0], verts[i][1], verts[i][2],
                            verts_color_0[0], verts_color_0[1], verts_color_0[2])

    for j in range(0, num_verts_2):
        verts_tuple[num_verts+j] = (verts_warped[j][0], verts_warped[j][1], verts_warped[j][2],
                            verts_color_1[0], verts_color_1[1], verts_color_1[2])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")

    ply_data = plyfile.PlyData([el_verts])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

def main_function(experiment_directory, data_source, continue_from, shapenet, sdf_mode):
    specs = ws.load_experiment_specifications(experiment_directory)
    train_split_file = specs["TrainSplit"]
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):
        ws.save_model(experiment_directory, "latest.pth", decoder, epoch)
        ws.save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        ws.save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):
        ws.save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        ws.save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        ws.save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"], sdf_mode=sdf_mode).cuda()
    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)
    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    num_samp_correspond = specs["SamplesPerSceneCorr"]
    if not shapenet:
        sdf_dataset = deep_sdf.data.SDFSamples(
            data_source, train_split_file, num_samp_per_scene, num_samp_correspond,
            load_ram=True, load_surface_points=True
        )
    else:
        with open(train_split_file, "r") as f:
            train_split = json.load(f)

        sdf_dataset = deep_sdf.data.ShapeNetSDFSamples(
            data_source, train_split, num_samp_per_scene,
            num_samp_correspond, load_ram=True, load_surface_points=True
        )

    if sdf_dataset.load_ram:
        num_data_loader_threads = 0
    else:
        num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    num_scenes = len(sdf_dataset)
    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
            lat_vecs.weight.data,
            0.0,
            get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
        )
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.module.warper.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": decoder.module.sdf_decoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
            {
                "params": decoder.module.sp_decoder.parameters(),
                "lr": lr_schedules[3].get_learning_rate(0),
            },
        ]
    )


    tensorboard_saver = ws.create_tensorboard_saver(experiment_directory)

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    loss_l1 = torch.nn.L1Loss(reduction="mean")
    loss_l1_soft = deep_sdf.loss.SoftL1Loss(reduction="mean")
    loss_l2 = torch.nn.MSELoss(reduction="mean")
    k = get_spec_with_default(specs, "k", 0.5)
    d = get_spec_with_default(specs, "d", 0.25)
    loss_lp = deep_sdf.loss.LipschitzLoss(k=k, reduction="sum")
    loss_sp = ComputeLoss3d()
    huber_fn = deep_sdf.loss.HuberFunc(reduction="mean")
    use_curriculum = get_spec_with_default(specs, "UseCurriculum", False)

    use_pointwise_loss = get_spec_with_default(specs, "UsePointwiseLoss", False)
    pointwise_loss_weight = get_spec_with_default(specs, "PointwiseLossWeight", 0.0)

    use_pointpair_loss = get_spec_with_default(specs, "UsePointpairLoss", False)
    pointpair_loss_weight = get_spec_with_default(specs, "PointpairLossWeight", 0.0)

    use_sdf_loss = get_spec_with_default(specs, "UseSdfLoss", False)
    sdf_loss_weight = get_spec_with_default(specs, "SdfLossWeight", 0.0)

    use_structural_loss = get_spec_with_default(specs, "UseStructuralLoss", False)
    structural_loss_weight = get_spec_with_default(specs, "StructuralLossWeight", 0.0)
    structural_loss_warp_weight = get_spec_with_default(specs, "StructuralPointsWarpWeight", 0.0)
    use_structural_warp_loss = get_spec_with_default(specs, "UseStructuralWarpLoss", False)
    structural_loss_shuffle_weight = get_spec_with_default(specs, "StructuralPointsShuffleWeight", 0.0)
    use_structural_shuffle_loss = get_spec_with_default(specs, "UseStructuralShuffleLoss", False)
    change_epoch = get_spec_with_default(specs, "ChangeEpoch", 0)
    use_input_sp = get_spec_with_default(specs, "UseInputSp", False)

    start_epoch = 1

    if continue_from is not None:
        if not os.path.exists(os.path.join(experiment_directory, ws.latent_codes_subdir, continue_from + ".pth")) or \
                not os.path.exists(os.path.join(experiment_directory, ws.model_params_subdir, continue_from + ".pth")) or \
                not os.path.exists(os.path.join(experiment_directory, ws.optimizer_params_subdir, continue_from + ".pth")):
            logging.warning('"{}" does not exist! Ignoring this argument...'.format(continue_from))
        else:
            logging.info('continuing from "{}"'.format(continue_from))

            lat_epoch = ws.load_latent_vectors(
                experiment_directory, continue_from + ".pth", lat_vecs
            )

            model_epoch = ws.load_model_parameters(
                experiment_directory, continue_from, decoder
            )

            optimizer_epoch = ws.load_optimizer(
                experiment_directory, continue_from + ".pth", optimizer_all
            )

            if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
                raise RuntimeError(
                    "epoch mismatch: {} vs {} vs {} ".format(
                        model_epoch, optimizer_epoch, lat_epoch
                    )
                )
            start_epoch = model_epoch + 1

            logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {},"
        "Number of warper parameters: {},"
        "Number of sdf decoder parameters: {},"
        "Number of sp decoder parameters: {},".format(
            sum(p.data.nelement() for p in decoder.parameters()),
            sum(p.data.nelement() for p in decoder.module.warper.parameters()),
            sum(p.data.nelement() for p in decoder.module.sdf_decoder.parameters()),
            sum(p.data.nelement() for p in decoder.module.sp_decoder.parameters()),
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    for epoch in range(start_epoch, num_epochs + 1):

        logging.info("epoch {}...".format(epoch))

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        batch_num = len(sdf_loader)
        for bi, data in enumerate(sdf_loader):
            decoder.train()
            out_data = data[0]
            on_data = data[1]
            indices = data[-1]
            #sdf_data = torch.cat([out_data[:, :, :3], on_data], dim=1)
            #sdf_data = out_data[:, :, :3].cuda()
            #sdf_data.requires_grad_(False)

            optimizer_all.zero_grad()

            batch_loss_sdf = 0.0
            batch_loss_pw = 0.0
            batch_loss_reg = 0.0
            batch_loss_pp = 0.0
            batch_loss = 0.0
            batch_loss_structure = 0.0
            batch_loss_structural_warp = 0.0
            batch_loss_structural_shuffle = 0.0
            # batch_loss_vol = 0.0

            batch_vecs = lat_vecs(indices).view(-1, latent_size).cuda()

            #input_xyz = sdf_data[:, :, :3].cuda()
            input_xyz = out_data[:, :, :3].cuda()
            input_xyz.requires_grad_(False)
            sdf_gt = out_data[:, :num_samp_per_scene, 3].unsqueeze(-1).cuda()
            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)
            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            warped_xyz_list, pred_sdf_list, input_sp, warped_sp = decoder(
                input_xyz, on_data[:, :, :3].cuda(), code_input=batch_vecs,
                output_warped_points=True, output_warping_param=True,
                predict_sp=True, use_input_sp=use_input_sp, mode='inverse')

            if enforce_minmax:
                for k in range(len(pred_sdf_list)):
                    pred_sdf_list[k] = torch.clamp(pred_sdf_list[k], minT, maxT)

            if use_sdf_loss:
                if use_curriculum:
                    sdf_loss = apply_curriculum_l1_loss(
                        pred_sdf_list, sdf_gt, loss_l1_soft, num_samp_per_scene)
                else:
                    sdf_loss = loss_l1(pred_sdf_list[-1][:, :num_samp_per_scene, :], sdf_gt)
                batch_loss_sdf += sdf_loss.item()
                chunk_loss = sdf_loss_weight * sdf_loss

            if do_code_regularization:
                reg_loss = torch.mean(torch.norm(batch_vecs, dim=1))
                chunk_loss = chunk_loss + code_reg_lambda * min(1.0, epoch / 100) * reg_loss.cuda()
                batch_loss_reg += reg_loss.item()

            warped_temp = warped_xyz_list[-1]
            warp_loss_input = torch.cat([input_xyz[:, :, :3].cuda(), on_data[:, :, :3].cuda()], dim=1)
            warp_loss_num = warp_loss_input.shape[1]
            if use_pointwise_loss:
                loss_func = huber_fn
                if use_curriculum:
                    pw_loss = apply_pointwise_reg(warped_xyz_list, warp_loss_input,
                                                  loss_func, warp_loss_num, d=d)
                else:
                    pw_loss = apply_pointwise_reg([warped_temp], warp_loss_input,
                                                  loss_func, warp_loss_num, d=d)
                chunk_loss += pw_loss.cuda() * pointwise_loss_weight * max(1.0, 10.0 * (1 - epoch / 100))
                batch_loss_pw += pw_loss.item()

            if use_pointpair_loss:
                lp_loss = apply_pointpair_reg(warped_temp, warp_loss_input,
                                              loss_lp, scene_per_batch)
                chunk_loss += lp_loss.cuda() * pointpair_loss_weight * min(1.0, epoch / 100)
                batch_loss_pp += lp_loss.item()

            if use_structural_loss:
                # init_pts_idx = farthest_point_sample(on_data[:, :, :3].cuda(), 128)
                # # init_pts = on_data[:, :, :3].cuda()[init_pts_idx]
                # init_pts = index_points(on_data[:, :, :3].cuda(), init_pts_idx)
                # structural_loss = apply_sp_loss(loss_sp, input_sp, init_pts)
                structural_loss = apply_sp_loss(loss_sp, input_sp, on_data[:, :, :3].cuda())
                chunk_loss += structural_loss.cuda() * structural_loss_weight
                batch_loss_structure += structural_loss.item()

                #
                if use_structural_warp_loss and epoch >= change_epoch:
                    #assume pts are in order:
                    structural_warp_extend = warped_sp.repeat(scene_per_batch, 1, 1)
                    structural_warp_shuffled = [warped_sp[i, :, :].repeat(scene_per_batch, 1, 1) for i in
                                                range(scene_per_batch)]
                    structural_warp_shuffled = torch.cat(structural_warp_shuffled, dim=0)
                    structural_warp_loss = loss_l2(structural_warp_extend, structural_warp_shuffled)
                    chunk_loss += structural_warp_loss.cuda() * structural_loss_warp_weight
                    batch_loss_structural_warp += structural_warp_loss.item()

                if use_structural_shuffle_loss and epoch >= change_epoch:
                    structural_warp_extend = warped_sp.repeat(scene_per_batch, 1, 1)
                    structural_warp_shuffled = [warped_sp[i, :, :].repeat(scene_per_batch, 1, 1) for i in
                                                range(scene_per_batch)]
                    structural_warp_shuffled = torch.cat(structural_warp_shuffled, dim=0)
                    dist1, dist2 = compute_chamfer_distance(structural_warp_shuffled, structural_warp_extend)
                    structural_shuffle_loss = torch.mean(dist1) + torch.mean(dist2)
                    batch_loss_structural_shuffle += structural_shuffle_loss.item()
                    chunk_loss += structural_loss_shuffle_weight * structural_shuffle_loss.cuda()

            chunk_loss.backward()
            batch_loss += chunk_loss.item()
            logging.info(
                "chunk_loss = {:.6f}, sdf_loss = {:.6f}, reg_loss = {:.6f},"
                "pw_loss = {:.6f}, pp_loss = {:.6f}, "
                "s_loss = {:.6f},s_shuffle_loss = {:.6f},"
                "s_warp_loss = {:.6f}".format(
                    chunk_loss.item(), batch_loss_sdf, batch_loss_reg, batch_loss_pw,
                    batch_loss_pp, batch_loss_structure,
                    batch_loss_structural_shuffle,
                    batch_loss_structural_warp))

            ws.save_tensorboard_logs(
                tensorboard_saver, epoch * batch_num + bi,
                loss_sdf=batch_loss_sdf, loss_pw=batch_loss_pw,
                loss_pp=batch_loss_pp,
                loss_s=batch_loss_structure,
                loss_s_shuffle=batch_loss_structural_shuffle)

            optimizer_all.step()

            del warped_xyz_list, pred_sdf_list, \
                reg_loss, batch_loss_reg,\
                batch_loss_sdf, batch_loss_pw, batch_loss, chunk_loss, \
                batch_loss_structure, batch_loss_structural_shuffle
            if use_pointwise_loss:
                del pw_loss
            if use_pointpair_loss:
                del lp_loss
            if use_structural_loss:
                del structural_loss, input_sp, warped_sp
                if use_structural_warp_loss and epoch >= change_epoch:
                    del structural_warp_loss, batch_loss_structural_warp
                if use_structural_shuffle_loss and epoch >= change_epoch:
                    del structural_shuffle_loss
            if use_sdf_loss:
                del sdf_loss

            if epoch in checkpoints:
                save_checkpoints(epoch)

            if epoch % log_frequency == 0:

                save_latest(epoch)
                ws.save_logs(
                    experiment_directory,
                    loss_log,
                    lr_log,
                    timing_log,
                    lat_mag_log,
                    param_mag_log,
                    epoch,
                )

if __name__ == "__main__":
    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)
    torch.cuda.manual_seed_all(31359)
    torch.backends.cudnn.deterministic = True

    import argparse
    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )

    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
             + "from the latest running snapshot, or an integer corresponding to "
             + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--shapenet",
        action='store_true',
        help="do not use corresponding points",
    )

    arg_parser.add_argument(
        "--sdf_mode",
        type=str,
        default='pointnet_nvp',
        help="do not use corresponding points",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.data_source, args.continue_from, args.shapenet, args.sdf_mode)
