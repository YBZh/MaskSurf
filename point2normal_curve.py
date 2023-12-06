import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# import numpy as np
# from pytorch3d.io import load_obj
# from pytorch3d.structures import Meshes
# from pytorch3d.ops import sample_points_from_meshes
# import torch
# import ipdb
# from pyntcloud import PyntCloud
# import pandas as pd
from PIL import Image, ImageFile

# from mesh_to_sdf import sample_sdf_near_surface, mesh_to_sdf
import trimesh
# import pyrender
# import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import numpy as np
import warnings
import pickle
import open3d
from open3d import *
from tqdm import tqdm
from torch.utils.data import Dataset
import ipdb


warnings.filterwarnings('ignore')

shapenet_v2_path = '/home/ssddata/yabin/shapenet/ShapeNetCore.v2/'
save_path = '/home/ssddata/yabin/shapenet/shape_deepsdf_pc_sdf/'


def mesh_normalize(mesh):
    v = mesh.vertices
    vmin = v.min(0)
    vmax = v.max(0)
    v = (v - vmin) / (vmax - vmin) * 2 - 1
    mesh.vertices = v
    return mesh

def to_unit_cube(mesh: trimesh.Trimesh):

    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0/bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)

    return mesh

## not used in data processing
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids



def compute_norm_and_curvature(pc, knn_indices=None):
    if knn_indices is not None:
        pc = pc[knn_indices]
    covariance = np.cov(pc.T)
    w, v = np.linalg.eig(covariance)
    v = v.T
    w = np.real(w)
    i = np.argmin(np.abs(w))
    norm = v[i]
    curv = w[i] / np.sum(np.abs(w))
    # assert curv is not complex
    return norm, np.real(curv)

shapenet_path = shapenet_v2_path
def extract_point_sdf(obj_path, point_number=125000, point_distribution='uniform'):
    # ipdb.set_trace()
    point = np.load(obj_path)
    # point, idx = mesh_norm_jiehong.sample(point_number, return_index=True)
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(point)
    kdtree = open3d.geometry.KDTreeFlann()
    kdtree.set_geometry(point_cloud)
    norms = []
    curvs = []
    for j in range(point.shape[0]):
        q = point[j]
        q = np.float64(q)
        k, indices, dist = kdtree.search_knn_vector_3d(q, knn=32)
        indices = np.asarray(indices)
        # print(indices.shape)
        norm, curv = compute_norm_and_curvature(point, indices)
        norms.append(norm)
        curvs.append(curv)
    norms = np.array(norms)
    curvs = np.array(curvs).reshape(point.shape[0], 1)
    # print(norms[:10])
    # print(curvs[:10])
    return np.concatenate((point, norms, curvs), 1)

    ### this is the official code to calculate the normal and curvs, however, there is some error for some points.
    # point, idx = mesh_norm_jiehong.sample(point_number, return_index=True)
    # norms = mesh_norm_jiehong.face_normals[idx]
    # curvs = trimesh.curvature.discrete_mean_curvature_measure(mesh_norm_jiehong, point, 0.2)
    # curvs = curvs.reshape(-1, 1)
    # return np.concatenate((point, norms, curvs), 1)


    # d = {'x': points_norm_jiehong[:, 0], 'y': points_norm_jiehong[:, 1], 'z': points_norm_jiehong[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # cloud.to_file("points_norm_jiehong.ply")

    ############################# trimesh.sample.sample_surface 也是从原本的mesh 上采点的，和mesh 的可视化是匹配的; 但是trimesh.sample_sdf_near_surface 是先做了to_unit_cube, 再点采样的。
    # points, idx = trimesh.sample.sample_surface(mesh, 250000)
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # cloud.to_file("vanilla_meash_sample.ply")

    # points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # cloud.to_file("vanilla.ply")



class ShapeNetDataLoader(Dataset):
    def __init__(self, shapenet_path='/home/ssddata/yabin/shapenet/ShapeNetCore.v2/', save_path='/home/ssddata/yabin/shapenet/shape_uniform_pc_sdf/', point_number=125000, point_distribution='uniform'):
        self.shapenet_path = shapenet_path
        self.save_path = save_path
        # os.makedirs(self.save_path)
        self.point_number = point_number
        self.point_distribution = point_distribution
        obj_path_list = []
        save_path_list = []
        ind = 0
        instance_list = os.listdir(shapenet_path)
        for instnace_name in instance_list:
            # print(ind)
            ind = ind + 1
            file_path = os.path.join(shapenet_path, instnace_name)
            save_file_name = os.path.join(save_path, instnace_name)
            # obj_file_name = os.path.join(class_path, instance_name, 'models', 'model_normalized.obj')
            obj_path_list.append(file_path)
            save_path_list.append(save_file_name)
        self.obj_path_list = obj_path_list
        self.save_path_list = save_path_list

    def __len__(self):
        return len(self.obj_path_list)

    def _get_item(self, index):
        print(index)
        save_file_name = self.save_path_list[index]
        obj_file_name = self.obj_path_list[index]
        if os.path.isfile(save_file_name):
            return np.array([0])
        else:
            # print(obj_file_name)
            ### 如果save file name 存在，则直接跳过？ 看看是哪个文件出了问题。
            point_cloud = extract_point_sdf(obj_file_name, point_number=self.point_number, point_distribution=self.point_distribution)
            np.save(save_file_name, point_cloud)
            return obj_file_name

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch
    shapenet_path = '/disk1/yabin/MaskSurf_data/ShapeNet55-34/shapenet_pc'

    save_path = '/disk1/yabin/MaskSurf_data/ShapeNet55-34/shapenet_pc_pointbert_estimated_normal_curve'
    point_number = 8192
    point_distribution = 'xyznormalcurvs'

    data = ShapeNetDataLoader(shapenet_path=shapenet_path, save_path=save_path, point_number=point_number, point_distribution=point_distribution)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=64)
    for i, (input) in enumerate(DataLoader):
        print(i)









# def npy2ply(obj_path, with_normal=False, point_number=8192):
#     points = np.load(obj_path)
#     d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
#     cloud = PyntCloud(pd.DataFrame(data=d))
#     cloud.to_file("04379243-afda402f59a7737ad11ab08e7440.ply")
#     # the returned normal is bad. [0, 1, 0], 只有0和1， 不知道为啥; 擦，这是个标准的方形，所以normal 还真可能是只有0,1
# obj_file_name = './04379243-afda402f59a7737ad11ab08e7440.npy'
# npy2ply(obj_file_name)
#


# mv 02747177-3a982b20a1c8ebf487b2ae2815c9.npy 02747177-.npy
# mv 04379243-afda402f59a7737ad11ab08e7440.npy 04379243-.npy
# 02747177-.npy is identical to 02747177-3a982b20a1c8ebf487b2ae2815c9.npy
# 04379243-.npy is identical to 04379243-afda402f59a7737ad11ab08e7440.npy

### 目前网上公开的shapenet 的点云是有问题的，少了两个文件，而且文件名称也有两个不合规范。
# 02747177-.npy is identical to 02747177-3a982b20a1c8ebf487b2ae2815c9.npy
# 04379243-.npy is identical to 04379243-afda402f59a7737ad11ab08e7440.npy
# the 04379243-114d3d770d9203fbec82976a49dc.npy, 04379243-8594658920d6ea7b23656ce81843.npy are not included in the public shapenet55.


















# ### This file is from ChaoZheng Wu.
#
#
# import numpy as np
# import os
# import trimesh
# import argparse
# import open3d
# from open3d import *
#
#
# def getDir(path):
#     fList = os.listdir(path)
#     F = []
#     for i in fList:
#         d = os.path.join(path,i)
#         if os.path.isdir(d):
#             F.append(os.path.join(i, 'train'))
#             F.append(os.path.join(i, 'test'))
#     return F
#
#
# def getFilenames(path, shapeName):
#     Dir = os.path.join(path, shapeName)
#     fList = os.listdir(Dir)
#     files = []
#     for i in fList:
#         f = os.path.join(Dir, i)
#         if os.path.isfile(f) and i.split('.')[-1] == 'off':
#             files.append(i)
#     return files
#
#
# def readFile(path, shapeName, fileName):
#     path = os.path.join(path, shapeName, fileName)
#     mesh = trimesh.load_mesh(path)
#     return mesh
#
# # def mesh_normalize(mesh):
# #     v = mesh.vertices
# #     vmin = v.min(0)
# #     vmax = v.max(0)
# #     v = (v - vmin) / (vmax - vmin) * 2 - 1
# #     mesh.vertices = v
# #     return mesh
#
# # def pc_normalize(pc):
# #     l = pc.shape[0]
# #     centroid = np.mean(pc, axis=0)
# #     pc = pc - centroid
# #     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
# #     pc = pc / m
#     # return pc
#
# def mesh_normalize(mesh):
#
#     v = mesh.vertices
#     l = v.shape[0]
#     centroid = np.mean(v, axis=0)
#     v = v - centroid
#     m = np.max(np.sqrt(np.sum(v**2, axis=1)))
#     v = v / m
#     mesh.vertices = v
#     return mesh
#
# def saveFile(data, path, fileName):
#     fileName = fileName.split('.')[0] + '.txt'
#     path = os.path.join(path, fileName)
#     np.savetxt(path, data, fmt='%.6f', delimiter=',')
#
#
# def modify_off(path, shapeName, fileName):
#     path = os.path.join(path, shapeName, fileName)
#     file = open(path, 'r+')
#     content = file.read()
#     #pos = content.find('OFF')
#     #content = content[:pos]+'\n'+content[pos:]
#     file.seek(0)
#     file.truncate()
#     #file = open(path,'w')
#     #print(content[:10])
#     #print(pos)
#     file.write(content.replace('OFF','OFF\n'))
#     file.close()
#
#

#
#
# def main():
#     pass
#
# parser = argparse.ArgumentParser(description=' ... ')
# parser.add_argument('--s', default=0, type=int, metavar='DIR', help='start shape')
# parser.add_argument('--t', default=None, type=int, metavar='DIR', help='end shape')
# parser.add_argument('--log', default='./log1.txt', type=str, metavar='DIR', help='log file')
#
#
# if __name__ == '__main__':
#     cfg  = parser.parse_args()
#     root = '/home/sharedData/mesh/align_datasets/modelnet40_manually_aligned'
#     save_root = '/home/sharedData/mesh/align_datasets/modelnet40_point_norm_curv_from_mesh'
#     save_root1 = '/home/sharedData/mesh/align_datasets/modelnet40_point_norm_curv_from_calculation'
#     Dir = getDir(root)
#     Dir.sort()
#     log = open(cfg.log, 'w+')
#     if cfg.t is not None:
#         Dir = Dir[cfg.s:cfg.t]
#     else:
#         Dir = Dir[cfg.s:]
#     print(Dir)
#     C_MIN = []
#     C_MAX = []
#     for d in Dir:
#         # if d.split('/')[0] == 'sofa':
#         #     continue
#         files = getFilenames(root, d)
#         print(d)
#         for f in files:
#             print(f)
#             exist_file = os.path.join(save_root, str(10000), d, f.split('.')[0]+'.txt')
#             if os.path.exists(exist_file):
#                 continue
#             try:
#                 mesh = readFile(root, d, f)
#             except:
#                 modify_off(root, d, f)
#                 mesh = readFile(root, d, f)
#             mesh = mesh_normalize(mesh)
#
#             # for i in [1024, 10000]:
#             i = 10000
#             print(i)
#             try:
#                 point, idx = mesh.sample(i, return_index=True)
#                 norms = mesh.face_normals[idx]
#                 # curvs = trimesh.curvature.discrete_mean_curvature_measure(mesh, point, 0.2)
#                 # curvs = curvs.reshape(-1, 1)
#             except:
#                 log.write(os.path.join(d, f)+'\n')
#                 continue
#             # assert point.shape[0] == curvs.shape[0]
#             assert point.shape == norms.shape
#             concat = np.concatenate([point, norms], -1)
#             savePath = os.path.join(save_root, str(i), d)
#             if not os.path.exists(savePath):
#                 os.makedirs(savePath)
#             saveFile(concat, savePath, f)
#
#     #         point_cloud = PointCloud()
#     #         point_cloud.points = Vector3dVector(point)
#     #         kdtree = KDTreeFlann()
#     #         kdtree.set_geometry(point_cloud)
#     #         norms = []
#     #         curvs = []
#     #         for j in range(point.shape[0]):
#     #             q = point[j]
#     #             q = np.float64(q)
#     #             k, indices, dist = kdtree.search_knn_vector_3d(q, knn=32)
#     #             indices = np.asarray(indices)
#     #             # print(indices.shape)
#     #             norm, curv = compute_norm_and_curvature(point, indices)
#     #             norms.append(norm)
#     #             curvs.append(curv)
#     #         norms = np.array(norms)
#     #         curvs = np.array(curvs).reshape(point.shape[0], 1)
#     #         C_MIN.append(curvs.min())
#     #         C_MAX.append(curvs.max())
#     #         concat = np.concatenate([point, norms, curvs], 1)
#     #         savePath = os.path.join(save_root1, str(i), d)
#     #         if not os.path.exists(savePath):
#     #             os.makedirs(savePath)
#     #         saveFile(concat, savePath, f)
#     # MIN = min(C_MIN)
#     # MAX = max(C_MAX)
#     # log.write('curv min:'+str(MIN))
#     # log.write('curv max:'+str(MAX))
#     # log.close()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
