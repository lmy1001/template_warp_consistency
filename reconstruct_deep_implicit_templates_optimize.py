#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np

import deep_sdf
import deep_sdf.workspace as ws
import torch.utils.data as data_utils
import trimesh

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    on_data,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram_ShapeNet(
            test_sdf, num_samples
        ).cuda()

        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.view(1, latent_size)
        xyz = xyz.view(1, -1, 3)
        pred_sdf = decoder(xyz, on_data, code_input=latent_inputs, predict_sp=True, mode='inverse')

        if e == 0:
            pred_sdf = decoder(xyz, on_data, code_input=latent_inputs, predict_sp=True, mode='inverse')

        num_points = xyz.shape[1]
        pred_sdf = pred_sdf[:, :num_points, :]
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.item())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.item()

    return loss_num, latent

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--seed",
        dest="seed",
        default=10,
        help="random seed",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        default=256,
        help="Marching cube resolution.",
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

    use_octree_group = arg_parser.add_mutually_exclusive_group()
    use_octree_group.add_argument(
        '--octree',
        dest='use_octree',
        action='store_true',
        help='Use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )
    use_octree_group.add_argument(
        '--no_octree',
        dest='use_octree',
        action='store_false',
        help='Don\'t use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"], sdf_mode=args.sdf_mode)

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()


    num_samp_correspond = specs["SamplesPerSceneCorr"]
    use_correspond_only = False
    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = 1
    '''
    if not args.shapenet:
        sdf_dataset = deep_sdf.data.SDFSamples(
            args.data_source, args.split_filename, num_samp_per_scene, num_samp_correspond,
            load_ram=True
        )
    else:
        with open(args.split_filename, "r") as f:
            train_split = json.load(f)

        sdf_dataset = deep_sdf.data.ShapeNetSDFSamples(
            args.data_source, train_split, num_samp_per_scene,
            num_samp_correspond, load_ram=True, load_surface_points=True
        )

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )
    '''
    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    #latent_vectors = ws.load_pre_trained_latent_vectors(args.experiment_directory, args.checkpoint)

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    clamping_function = None
    if specs["NetworkArch"] == "deep_sdf_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])
    elif specs["NetworkArch"] == "deep_implicit_template_decoder":
        # clamping_function = lambda x: x * specs["ClampingDistance"]
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    if args.shapenet:
        with open(args.split_filename, "r") as f:
            train_split = json.load(f)
        npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, train_split)
    else:
        npz_filenames = deep_sdf.data.get_instance_filenames_dfaust(args.data_source, args.split_filename)

    # random.shuffle(npz_filenames)
    npz_filenames = sorted(npz_filenames)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)
        on_data_filename = os.path.join(args.data_source, ws.surface_samples_subdir, npz[:-4]+".ply")
        on_data_mesh = trimesh.load(on_data_filename)
        logging.debug("loading {}".format(npz))

        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
        on_data = trimesh.sample.sample_surface(on_data_mesh, num_samp_correspond)[0]
        on_data = torch.from_numpy(on_data).float().view(1, -1, 3).cuda()

        #code_input = latent_vectors[bi].view(1, -1).cuda()
        mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
        latent_filename = os.path.join(
            reconstruction_codes_dir, npz[:-4] + ".pth"
        )
        if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
        ):
            continue

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))

        if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
        ):
            continue

        logging.info("reconstructing {}".format(npz))

        data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

        start = time.time()
        if not os.path.isfile(latent_filename):
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                on_data,
                0.001,  # [emp_mean,emp_var],
                0.1,
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
            )
            logging.info("reconstruct time: {}".format(time.time() - start))
            logging.info("reconstruction error: {}".format(err))
            err_sum += err
            logging.info("current_error avg: {}".format((err_sum / (ii + 1))))
            # logging.debug(ii)

            # logging.debug("latent: {}".format(latent.detach().cpu().numpy()))
        else:
            logging.info("loading from " + latent_filename)
            latent = torch.load(latent_filename).view(1, -1).cuda()

        if not save_latvec_only:
            start = time.time()
            with torch.no_grad():
                if args.use_octree:
                    deep_sdf.mesh.create_mesh_octree(
                        decoder, on_data, latent, mesh_filename, N=args.resolution, max_batch=int(2 ** 17),
                        clamp_func=clamping_function
                    )
                else:
                    deep_sdf.mesh.create_mesh(
                        decoder, on_data, latent, mesh_filename, N=args.resolution, max_batch=int(2 ** 17),
                    )
            logging.debug("total time: {}".format(time.time() - start))

        if not os.path.exists(os.path.dirname(latent_filename)):
            os.makedirs(os.path.dirname(latent_filename))

        torch.save(latent.unsqueeze(0), latent_filename)
