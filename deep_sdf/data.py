#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import tqdm

import deep_sdf.workspace as ws
import trimesh
import yaml

def get_instance_filenames_dfaust(data_source, split):
    npzfiles = []
    categories = ['DFaust_data']
    # Read metadata file
    metadata_file = os.path.join(data_source, 'metadata.yaml')

    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = yaml.load(f)
    else:
        metadata = {
            c: {'id': c, 'name': 'n/a'} for c in categories
        }

    # Get all models
    models = []
    for c_idx, c in enumerate(categories):
        subpath = os.path.join(data_source, ws.sdf_samples_subdir, c)
        if not os.path.isdir(subpath):
            logging.debug('Category %s does not exist in dataset.' % c)
        if os.path.exists(split):
            split_file = split
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
        else:
            models_c = [f for f in os.listdir(subpath) if
                        os.path.isdir(os.path.join(subpath, f))]
        models_c = list(filter(lambda x: len(x) > 0, models_c))
        instance_names = []
        for idx, model in enumerate(models_c):
            #print("model: ", model)
            folder = os.path.join(subpath, model)
            files = os.listdir(folder)
            files.sort()
            #files = glob.glob(os.path.join(folder, '*.npz'))
            #files.sort()
            for npz_file in files:
                instance_filename = os.path.join(
                    c, model, npz_file
                )
                #print("instance_filename: ", instance_filename)
                if not os.path.isfile(
                        os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)
    randidx = torch.randperm(samples.shape[0])
    samples = torch.index_select(samples, 0, randidx)

    return samples


def unpack_sdf_samples_from_ram(data, gt_tensor, subsample=None, num_correspond=None, gt_order_idx=None):
    if subsample is None:
        return data, None

    pos_tensor = data[0]
    neg_tensor = data[1]
    on_tensor = data[2]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)
    randidx = torch.randperm(samples.shape[0])
    samples = torch.index_select(samples, 0, randidx)

    if num_correspond:
        on_tensor_idx = np.random.randint(on_tensor.shape[0], size=(num_correspond))
        on_data = on_tensor[on_tensor_idx, :]
        if gt_order_idx is None:
            gt_data = gt_tensor[on_tensor_idx, :]
        else:
            gt_data = on_tensor[gt_order_idx, :]

    return samples, on_data, gt_data


def unpack_sdf_samples_in_order(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    sample_pos = pos_tensor[:half, :]
    sample_neg = neg_tensor[:half, :]

    samples = torch.cat([sample_pos, sample_neg], 0)
    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        sample_correspond,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        use_correspond_only=False,
        evaluation=False,
        normalize=True,
        load_surface_points=True,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames_dfaust(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        self.load_surface_points = load_surface_points
        if load_ram:
            self.loaded_data = []
            if self.load_surface_points:
                self.surface_data = []
                self.offset_data = []
                self.scale_data = []
            for f in tqdm.tqdm(self.npyfiles, ascii=True):
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )
                if self.load_surface_points:
                    surfacefile = os.path.join(self.data_source, ws.surface_samples_subdir, f[:-4] + ".ply")
                    surface_file = trimesh.load(surfacefile)
                    surface_points = torch.from_numpy(np.array(surface_file.vertices).astype(np.float32))
                    # ridx = np.random.choice(surface_points.shape[0], self.on_nums)
                    # surface_points = surface_points[ridx, :]
                    self.surface_data.append(surface_points)

                    normalization_filename = os.path.join(self.data_source, ws.normalization_param_subdir, f)
                    normalization_params = np.load(normalization_filename)
                    self.offset_data.append(normalization_params["offset"])
                    self.scale_data.append(normalization_params["scale"])

        self.on_nums = sample_correspond
        self.normalize = normalize

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        # filename = os.path.join(
        #     self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        # )

        # normalization_params = np.load(
        #     os.path.join(self.data_source, ws.normalization_param_subdir, self.npyfiles[idx]))
        # offset = normalization_params["offset"]
        # scale = normalization_params["scale"]
        offset = self.offset_data[idx]
        scale = self.scale_data[idx]

        samples = unpack_sdf_samples_from_ram_ShapeNet(
            self.loaded_data[idx], self.subsample)
        # if self.normalize:
        #    samples[:, :3] = (samples[:, :3] + offset) * scale #- offset

        if self.load_surface_points:
            # surface points
            # surfacefile = os.path.join(
            #     self.data_source,
            #     ws.surface_samples_subdir, self.npyfiles[idx][:-4] + ".ply")
            # surface_file = trimesh.load(surfacefile)
            # surface_points = torch.from_numpy(np.array(surface_file.vertices).astype(np.float32))
            # ridx = np.random.choice(surface_points.shape[0], self.on_nums)
            # surface_points = surface_points[ridx, :]
            surface_points = self.surface_data[idx]
            ridx = np.random.choice(surface_points.shape[0], self.on_nums)
            surface_points = surface_points[ridx, :]

            if self.normalize:
                surface_points[:, :3] = (surface_points[:, :3] + offset) * scale  # - offset

            return samples, surface_points, idx
        else:
            return samples, idx


def unpack_sdf_samples_from_ram_ShapeNet(data, subsample=None):
    if subsample is None:
        return data, None

    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)
    randidx = torch.randperm(samples.shape[0])
    samples = torch.index_select(samples, 0, randidx)
    return samples

class ShapeNetSDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        sample_correspond=5000,
        load_ram=False,
        load_surface_points=False,
        normalize=True
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        self.on_nums = sample_correspond
        self.load_surface_points = load_surface_points
        self.normalize = normalize

        if load_ram:
            self.loaded_data = []
            if self.load_surface_points:
                self.surface_data = []
                self.offset_data = []
                self.scale_data = []
            for f in tqdm.tqdm(self.npyfiles, ascii=True):
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )
                if self.load_surface_points:
                    surfacefile = os.path.join(self.data_source, ws.surface_samples_subdir, f[:-4] + ".ply")
                    surface_file = trimesh.load(surfacefile)
                    surface_points = torch.from_numpy(np.array(surface_file.vertices).astype(np.float32))
                    # ridx = np.random.choice(surface_points.shape[0], self.on_nums)
                    # surface_points = surface_points[ridx, :]
                    self.surface_data.append(surface_points)

                    normalization_filename = os.path.join(self.data_source, ws.normalization_param_subdir, f)
                    normalization_params = np.load(normalization_filename)
                    self.offset_data.append(normalization_params["offset"])
                    self.scale_data.append(normalization_params["scale"])


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        # filename = os.path.join(
        #     self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        # )

        offset = self.offset_data[idx]
        scale = self.scale_data[idx]

        samples = unpack_sdf_samples_from_ram_ShapeNet(
            self.loaded_data[idx], self.subsample)

        if self.load_surface_points:
            surface_points = self.surface_data[idx]
            ridx = np.random.choice(surface_points.shape[0], self.on_nums)
            surface_points = surface_points[ridx, :]
            if self.normalize:
                surface_points[:, :3] = (surface_points[:, :3] + offset) * scale #- offset
            return samples, surface_points, idx
        else:
            return samples, idx


