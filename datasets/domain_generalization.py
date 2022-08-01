'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch


import torch.utils.data as data
import sys
import h5py
import numpy as np
import glob
import ipdb

warnings.filterwarnings('ignore')


import numpy as np

def normal_pc(pc):
    """
    normalize point cloud in range L
    :param pc: type list
    :return: type list
    """
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc/pc_L_max
    return pc

idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}

def rotation_point_cloud(pc):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    # rotation_matrix = np.array([[cosval, 0, sinval],
    #                             [0, 1, 0],
    #                             [-sinval, 0, cosval]])
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    # rotation_matrix = np.array([[cosval, -sinval, 0],
    #                             [sinval, cosval, 0],
    #                             [0, 0, 1]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud_by_angle(pc, rotation_angle):
    """
    Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction
    :param pc: B X N X 3 array, original batch of point clouds
    :param rotation_angle: angle of rotation
    :return: BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)

    # rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data

def random_rotate_one_axis(X, axis):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    return X.astype('float32')

def rotate_shape(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if axis == "x":
        return x.dot(R_x).astype('float32')
    elif axis == "y":
        return x.dot(R_y).astype('float32')
    else:
        return x.dot(R_z).astype('float32')

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud.astype('float32')

def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter points. jittering is per point.
    :param pc: B X N X 3 array, original batch of point clouds
    :param sigma:
    :param clip:
    :return:
    """
    jittered_data = np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip)
    jittered_data += pc
    return jittered_data


def shift_point_cloud(pc, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    pc += shifts
    return pc


def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
    """
    N, C = pc.shape
    scales = np.random.uniform(scale_low, scale_high, 1)
    pc *= scales
    return pc


def rotate_perturbation_point_cloud(pc, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(pc.shape, dtype=np.float32)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    shape_pc = pc
    rotated_data = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def pc_augment(pc):
    pc = rotation_point_cloud(pc)
    pc = jitter_point_cloud(pc)
    # pc = random_scale_point_cloud(pc)
#    pc = rotate_perturbation_point_cloud(pc)
    # pc = shift_point_cloud(pc)
    return pc

def load_dir(data_dir, name='train_files.txt'):
    with open(os.path.join(data_dir,name),'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]


def get_info(shapes_dir, isView=False):
    names_dict = {}
    if isView:
        for shape_dir in shapes_dir:
            name = '_'.join(os.path.split(shape_dir)[1].split('.')[0].split('_')[:-1])
            if name in names_dict:
                names_dict[name].append(shape_dir)
            else:
                names_dict[name] = [shape_dir]
    else:
        for shape_dir in shapes_dir:
            name = os.path.split(shape_dir)[1].split('.')[0]
            names_dict[name] = shape_dir

    return names_dict
####################3
# the above codes are cloned from Point DAN.
####################


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc




def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

@DATASETS.register_module()
class Modelnet40_DG(Dataset):
    def __init__(self, config):
    # def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True):
    #     super(Modelnet40_DG, self).__init__()
    #     self.status = status
        self.pc_list = []
        self.lbl_list = []
    #     self.pc_input_num = pc_input_num
    #     self.aug = aug
        self.root = config.DATA_PATH   ## data/PointDA_data/modelnet
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset

        categorys = glob.glob(os.path.join(self.root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        # sorted(categorys)
        categorys = sorted(categorys)

        if split == 'train':
            npy_list = glob.glob(os.path.join(self.root, '*', 'train', '*.npy'))
        else:
            npy_list = glob.glob(os.path.join(self.root, '*', 'test', '*.npy'))
        # names_dict = get_info(npy_list, isView=False)

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))

        print(f'{split} data num: {len(self.pc_list)}')

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        # print(lbl)
        pc = np.load(self.pc_list[idx])[:, :3].astype(np.float32)
        pc = normal_pc(pc)
        pc = farthest_point_sample(pc, self.npoints)
        pt_idxs = np.arange(0, pc.shape[0])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
            current_points = pc[pt_idxs].copy()
            current_points = random_rotate_one_axis(current_points, "z")
            current_points = jitter_pointcloud(current_points)
        else:
            current_points = pc[pt_idxs].copy()
            # pc = rotation_point_cloud(pc)
            # pc = jitter_point_cloud(pc)
        # print(pc.shape)
        # pc = np.expand_dims(pc.transpose(), axis=2)
        # return torch.from_numpy(pc).type(torch.FloatTensor), lbl

        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, lbl)

    def __len__(self):
        return len(self.pc_list)

@DATASETS.register_module()
class Shapenet_DG(data.Dataset):
    def __init__(self, config):
    # def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True, data_type='*.npy'):
    #     super(Shapenet_data, self).__init__()
        # self.status = status
        self.pc_list = []
        self.lbl_list = []
        self.root = config.DATA_PATH  ## data/PointDA_data/modelnet
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset
        # self.pc_input_num = pc_input_num
        # self.aug = aug
        self.data_type = '*.npy'

        categorys = glob.glob(os.path.join(self.root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        # sorted(categorys)
        categorys = sorted(categorys)

        if split == 'train':
            pts_list = glob.glob(os.path.join(self.root, '*', 'train', self.data_type))
        elif split == 'test':
            pts_list = glob.glob(os.path.join(self.root, '*', 'test', self.data_type))
        else:
            pts_list = glob.glob(os.path.join(self.root, '*', 'validation', self.data_type))
        # names_dict = get_info(pts_list, isView=False)

        for _dir in pts_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))

        print(f'{split} data num: {len(self.pc_list)}')

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = np.load(self.pc_list[idx])[:self.npoints].astype(np.float32)
        pc = normal_pc(pc)
        # print(lbl)
        pc = self.rotate_pc(pc, lbl)
        pc = farthest_point_sample(pc, self.npoints)
        pt_idxs = np.arange(0, pc.shape[0])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
            current_points = pc[pt_idxs].copy()
            current_points = random_rotate_one_axis(current_points, "z")
            current_points = jitter_pointcloud(current_points)
        else:
            current_points = pc[pt_idxs].copy()

        # pt_idxs = np.arange(0, pc.shape[0])
        # if self.subset == 'train':
        #     np.random.shuffle(pt_idxs)
        #     # pc = rotation_point_cloud(pc)
        #     # pc = jitter_point_cloud(pc)
        # # print(pc.shape)
        # # pc = np.expand_dims(pc.transpose(), axis=2)
        # # return torch.from_numpy(pc).type(torch.FloatTensor), lbl
        # current_points = pc[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ShapeNet', 'sample', (current_points, lbl)
        # lbl = self.lbl_list[idx]
        # if self.data_type == '*.pts':
        #     pc = np.array([[float(value) for value in xyz.split(' ')]
        #                    for xyz in open(self.pc_list[idx], 'r') if len(xyz.split(' ')) == 3])[:self.pc_input_num, :]
        # elif self.data_type == '*.npy':
        #     pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        # pc = normal_pc(pc)
        # if self.aug:
        #     pc = rotation_point_cloud(pc)
        #     pc = jitter_point_cloud(pc)
        # pad_pc = np.zeros(shape=(self.pc_input_num - pc.shape[0], 3), dtype=float)
        # pc = np.concatenate((pc, pad_pc), axis=0)
        # pc = np.expand_dims(pc.transpose(), axis=2)
        # return torch.from_numpy(pc).type(torch.FloatTensor), lbl

    # shpenet is rotated such that the up direction is the y axis in all shapes except plant
    def rotate_pc(self, pointcloud, label):
        if label != label_to_idx["plant"]:
            pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud

    def __len__(self):
        return len(self.pc_list)

@DATASETS.register_module()
class Scannet_DG(data.Dataset):
    def __init__(self, config):
        # def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True):
        #     super(Scannet_data_h5, self).__init__()
        #     self.num_points = pc_input_num
        #     self.status = status
        #     self.aug = aug
            # self.label_map = [2, 3, 4, 5, 6, 7, 9, 10, 14, 16]
        self.root = config.DATA_PATH  ## data/PointDA_data/modelnet
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset
        if split == 'train':
            data_pth = load_dir(self.root, name='train_files.txt')
        else:
            data_pth = load_dir(self.root, name='test_files.txt')

        point_list = []
        label_list = []
        for pth in data_pth:
            data_file = h5py.File(pth, 'r')
            point = data_file['data'][:]
            label = data_file['label'][:]

            # idx = [index for index, value in enumerate(list(label)) if value in self.label_map]
            # point_new = point[idx]
            # label_new = np.array([self.label_map.index(value) for value in label[idx]])

            point_list.append(point)
            label_list.append(label)
        self.data = np.concatenate(point_list, axis=0)
        self.label = np.concatenate(label_list, axis=0)
        # ipdb.set_trace()
        print(self.label)

    def __getitem__(self, idx):
        # point_idx = np.arange(0, self.npoints)
        # np.random.shuffle(point_idx)
        point = self.data[idx][:self.npoints].astype(np.float32)[:, :3]
        lbl = self.label[idx]
        pc = normal_pc(point)
        pc = self.rotate_pc(pc)
        pc = farthest_point_sample(pc, self.npoints)
        pt_idxs = np.arange(0, pc.shape[0])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
            current_points = pc[pt_idxs].copy()
            current_points = random_rotate_one_axis(current_points, "z")
            current_points = jitter_pointcloud(current_points)
        else:
            current_points = pc[pt_idxs].copy()
        # pt_idxs = np.arange(0, pc.shape[0])
        # if self.subset == 'train':
        #     np.random.shuffle(pt_idxs)
        #
        # current_points = pc[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'Scannet', 'sample', (current_points, lbl)
        # pc = np.expand_dims(pc.transpose(), axis=2)
        # return torch.from_numpy(pc).type(torch.FloatTensor), label

    # scannet is rotated such that the up direction is the y axis
    def rotate_pc(self, pointcloud):
        pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud

    def __len__(self):
        return self.data.shape[0]


