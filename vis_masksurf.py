import os
import numpy as np
from pytorch3d.io import load_obj, save_obj
from pytorch3d.io import IO
from pytorch3d.io.ply_io import _save_ply

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import torch
import math
import ipdb
# from pyntcloud import PyntCloud
# import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pyntcloud import PyntCloud
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from matplotlib import pyplot as plt, colors
# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath(''))

shapenet_v2_path = '/home/ssddata/shapenet/shapenet_v2/ShapeNetCore.v2/'
save_path = '/home/yabin/syn_project/point_cloud/Point-MAE/data/ShapeNet55-34/shapenet_pc_ours_with_normal/'


# ## 在数据处理的时候不！需要 （）
# def pc_normalize(pc):
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / m
#     return pc


from pytorch3d.renderer import TexturesVertex



def tri_with_center_and_normal(center, normal, r):
    normal = normal.float()
    random_vector = torch.tensor([0.5, 0.5, 0.5]).float() # 1 * 3
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)
    if (normal == random_vector).int().sum().item() == 0:
        random_vector = torch.tensor([1, 0, 0]).float() # 1 * 3
    perpendicular_vector = torch.cross(normal, random_vector, dim=-1)
    perpendicular_vector = perpendicular_vector / torch.norm(perpendicular_vector, dim=-1, keepdim=True)
    rotation1 = rotation_vector_around_vector(perpendicular_vector, normal, torch.tensor([math.pi * 2.0 / 3.0]))
    rotation2 = rotation_vector_around_vector(perpendicular_vector, normal, torch.tensor([math.pi * 4.0 / 3.0]))
    return center + perpendicular_vector * r, center + rotation1 * r, center + rotation2 * r

def rotation_vector_around_vector(input_vector, anchor_vector, rotation_degree):
    # input_vector: torch size 3
    c = torch.cos(rotation_degree)
    s = torch.sin(rotation_degree)
    C = 1.0 - c
    Q = torch.zeros(3, 3)
    Q[0, 0] = anchor_vector[0] * anchor_vector[0] * C + c
    Q[0, 1] = anchor_vector[1] * anchor_vector[0] * C - anchor_vector[2] * s
    Q[0, 2] = anchor_vector[2] * anchor_vector[0] * C + anchor_vector[1] * s

    Q[1, 0] = anchor_vector[1] * anchor_vector[0] * C + anchor_vector[2] * s
    Q[1, 1] = anchor_vector[1] * anchor_vector[1] * C + c
    Q[1, 2] = anchor_vector[2] * anchor_vector[1] * C - anchor_vector[0] * s

    Q[2, 0] = anchor_vector[0] * anchor_vector[2] * C - anchor_vector[1] * s
    Q[2, 1] = anchor_vector[2] * anchor_vector[1] * C + anchor_vector[0] * s
    Q[2, 2] = anchor_vector[2] * anchor_vector[2] * C + c

    # expand_input = input_vector.view(3, 1).expand(3,3)
    # return (expand_input * Q).sum(1)
    # output = torch.zeros(3)
    # output[0] = input_vector[0] * Q[0, 0] + input_vector[0] * Q[0,1] + input_vector[0] * Q[0,2]
    # output[1] = input_vector[1] * Q[1, 0] + input_vector[1] * Q[1, 1] + input_vector[1] * Q[1, 2]
    # output[2] = input_vector[2] * Q[2, 0] + input_vector[2] * Q[2, 1] + input_vector[2] * Q[2, 2]

    output = torch.matmul(Q.float(), input_vector.view(3,1).float())
    return output.view(3)



def generate_surfel_mesh(points, normals):
    # 1024 * 3.
    # for each points, calculate its three tri neighbor.
    verts = torch.rand(points.size(0) * 3, points.size(1))
    for i in range(points.size(0)):
        tri_one, tri_two, tri_thr = tri_with_center_and_normal(points[i], normals[i], r=0.005)
        verts[i * 3] = tri_one
        verts[i * 3 + 1] = tri_two
        verts[i * 3 + 2] = tri_thr

    faces = torch.rand(points.size())
    for i in range(points.size(0)):
        faces[i, 0] = i * 3
        faces[i, 1] = i * 3 + 1
        faces[i, 2] = i * 3 + 2
    faces = faces.long()

    return verts, faces

# perpendicular_vector = torch.tensor([0,1,0])
# normal = torch.tensor([0,0,1])
# rotation1 = rotation_vector_around_vector(perpendicular_vector, normal, torch.tensor([math.pi * 2.0 / 3.0]))
# rotation2 = rotation_vector_around_vector(perpendicular_vector, normal, torch.tensor([math.pi * 4.0 / 3.0]))
# print(rotation1)
# print(rotation2)


# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
R, T = look_at_view_transform(2.7, 0, 180)
cameras = FoVPerspectiveCameras( R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
# the difference between naive and coarse-to-fine rasterization.
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the
# -z direction.
lights = PointLights(location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
# interpolate the texture uv coordinates for each vertex, sample from a texture image and
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        cameras=cameras,
        lights=lights
    )
)

# shapenet_path = shapenet_v2_path
# def extract_point_normal(obj_path, with_normal=True, point_number=1024):
#     try:
#         verts, faces, _ = load_obj(obj_path)
#     except:
#         log = open(os.path.join('log.txt'), 'a')
#         log.write('\n-------------------------------------------\n')
#         log.write(obj_path)
#         log.close()
#         print(obj_path)
#         return np.array([0])
#
#     test_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
#     sample_test = sample_points_from_meshes(test_mesh, point_number, return_normals=with_normal)  # float tensor
#
#     verts, faces = generate_surfel_mesh(sample_test[0][0], sample_test[1][0])
#
#     # # assign constant color texture
#     color = torch.zeros(1, verts.size(0), 3).float()
#     color[:, :1000, 0] = 1
#     color[:, 1000:2000, 1] = 0
#     color[:, 2000:, 2] = 0
#     print(color.size())
#     print(color)
#     test_mesh = Meshes(verts=[verts], faces=[faces]) # textures=TexturesVertex(verts_features=color))
#     test_mesh.textures = TexturesVertex(verts_features=color)
#
#     IO().save_mesh(test_mesh, 'test.ply', binary=False, colors_as_uint8=True)

# point_cloud = extract_point_normal(obj_file_name)

def generate_surfel_cloud(points, normals, root, name, normal_angle):
    verts, faces = generate_surfel_mesh(points, normals)
    # # assign constant color texture
    color = torch.zeros(1, verts.size(0), 3).float()
    ## assign color accroding to normal angle.
    normal_angle[normal_angle > 30] = 80
    normal_angle[normal_angle < 30] = 0
    # normal_angle = normal_angle.int()
    cmap = plt.cm.cool
    norm = colors.Normalize(vmin=0.0, vmax=80.0)
    color = torch.from_numpy(cmap(norm(normal_angle)))[:,:3].repeat(1,3).reshape(1,-1, 3)
    # color[:, :1000, 0] = 1
    # color[:, 1000:2000, 1] = 0
    # color[:, 2000:, 2] = 0
    # print(color.size())
    # print(color)
    test_mesh = Meshes(verts=[verts], faces=[faces])  # textures=TexturesVertex(verts_features=color))
    test_mesh.textures = TexturesVertex(verts_features=color)
    IO().save_mesh(test_mesh, root + name + '_surfel.ply', binary=False, colors_as_uint8=True)


# Initialize a camera.
R_point, T_point = look_at_view_transform(20, 10, 0)
cameras_point = FoVOrthographicCameras(R=R_point, T=T_point, znear=0.01)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
raster_settings_point = PointsRasterizationSettings(
    image_size=512,
    radius = 0.003,
    points_per_pixel = 10
)


# Create a points renderer by compositing points using an alpha compositor (nearer points
# are weighted more heavily). See [1] for an explanation.
rasterizer_point = PointsRasterizer(cameras=cameras_point, raster_settings=raster_settings_point)
renderer_point = PointsRenderer(
    rasterizer=rasterizer_point,
    compositor=AlphaCompositor()
)

def generate_point_cloud(points, root, name, point_dis):
    color = torch.zeros(1, points.size(0), 3).float()
    ## assign color accroding to normal angle.
    # normal_angle[normal_angle > 30] = 30
    # normal_angle = normal_angle.int()
    cmap = plt.cm.cool
    norm = colors.Normalize(vmin=0.0, vmax=0.001)
    color = torch.from_numpy(cmap(norm(point_dis)))[:,:3].reshape(-1, 3) * 255
    color = color.long()
    # print(points.size())
    # print(color.size())
    # print(color)
    # df = pd.DataFrame(
    #     # same arguments that you are passing to visualize_pcl
    #     data=np.hstack((np.array(points), np.array(color))),
    #     columns=["x", "y", "z", "red", "green", "blue"])
    # df[['red', 'green', 'blue']] = df[['red', 'green', 'blue']].astype(np.uint)
    # point_cloud = Pointclouds(points=[points], features=[color])
    # IO().save_point_cloud(point_cloud, root + name + '_reconstruct_pc.obj')
    # image = renderer_point(point_cloud)
    # cloud = PyntCloud(df)
    # cloud.plot() ### we have colored points here.
    fout = open(root + name + '_reconstruct_pc.obj', 'w')
    for i in range(points.size(0)):
        fout.write('v %f %f %f %d %d %d\n' % (
            points[i, 0], points[i, 1], points[i, 2], color[i, 0], color[i, 1],
            color[i, 2]))
    fout.close()
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2], 'red': color[:,0], 'green': color[:,1], 'blue':color[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # cloud.to_file(root + name + '_reconstruct_pc.ply')
    # # assign constant color texture

    # color[:, :1000, 0] = 1
    # color[:, 1000:2000, 1] = 0
    # color[:, 2000:, 2] = 0
    # print(color.size())
    # print(color)
    # test_mesh = Meshes(verts=[verts], faces=[faces])  # textures=TexturesVertex(verts_features=color))
    # test_mesh.textures = TexturesVertex(verts_features=color)
    # IO().save_mesh(test_mesh, root + name + '_reconstruct_pc.ply', binary=False, colors_as_uint8=True)

def generate_point_cloud_full(points, root, name, point_dis):
    color = torch.zeros(1, points.size(0), 3).float()
    ## assign color accroding to normal angle.
    # normal_angle[normal_angle > 30] = 30
    # normal_angle = normal_angle.int()
    cmap = plt.cm.cool
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    color = torch.from_numpy(cmap(norm(point_dis)))[:,:3].reshape(-1, 3) * 255
    color = color.long()
    # print(points.size())
    # print(color.size())
    # print(color)
    # df = pd.DataFrame(
    #     # same arguments that you are passing to visualize_pcl
    #     data=np.hstack((np.array(points), np.array(color))),
    #     columns=["x", "y", "z", "red", "green", "blue"])
    # df[['red', 'green', 'blue']] = df[['red', 'green', 'blue']].astype(np.uint)
    # point_cloud = Pointclouds(points=[points], features=[color])
    # IO().save_point_cloud(point_cloud, root + name + '_reconstruct_pc.obj')
    # image = renderer_point(point_cloud)
    # cloud = PyntCloud(df)
    # cloud.plot() ### we have colored points here.
    d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2], 'red': color[:,0], 'green': color[:,1], 'blue':color[:, 2]}
    cloud = PyntCloud(pd.DataFrame(data=d))
    cloud.to_file(root + name + '_gt_pc.ply')

root = './vis/02691156_520/'
name = 'mae' # 'masksurl' #
obj_file_name = root + name + '.pth.tar'
a = torch.load(obj_file_name)
input_pc = a['input_vis_point']
output_surfels = a['output_surfels']
full_rebuild_points = output_surfels[:, :3]
full_rebuild_normal = output_surfels[:, 3:6]
point_dis = output_surfels[:, 6]
normal_angle = output_surfels[:, 7]
full_input = output_surfels[:, 8:]

# print(normal_angle.mean()) # mae, 28.91, masksurf:
# mae:     1216; 974, 828, 671, 532
# masksurl:1172; 930, 801, 647, 494
print((normal_angle>10).int().sum())
print((normal_angle>20).int().sum())
print((normal_angle>30).int().sum())
print((normal_angle>40).int().sum())
print((normal_angle>50).int().sum())

d = {'x': input_pc[:, 0], 'y': input_pc[:, 1], 'z': input_pc[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
cloud.to_file(root + name + '_inputpc.ply')
#
generate_point_cloud(full_rebuild_points, root, name, point_dis)
generate_point_cloud_full(full_input, root, name, point_dis)
generate_surfel_cloud(full_rebuild_points, full_rebuild_normal, root, name, normal_angle)

name = 'masksurf' #
obj_file_name = root + name + '.pth.tar'
a = torch.load(obj_file_name)
input_pc = a['input_vis_point']
output_surfels = a['output_surfels']
full_rebuild_points = output_surfels[:, :3]
full_rebuild_normal = output_surfels[:, 3:6]
point_dis = output_surfels[:, 6]
normal_angle = output_surfels[:, 7]
full_input = output_surfels[:, 8:]

# print(normal_angle.mean()) # mae, 28.91, masksurf:
# mae:     1216; 974, 828, 671, 532
# masksurl:1172; 930, 801, 647, 494
print((normal_angle>10).int().sum())
print((normal_angle>20).int().sum())
print((normal_angle>30).int().sum())
print((normal_angle>40).int().sum())
print((normal_angle>50).int().sum())

d = {'x': input_pc[:, 0], 'y': input_pc[:, 1], 'z': input_pc[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
cloud.to_file(root + name + '_inputpc.ply')
#
generate_point_cloud(full_rebuild_points, root, name, point_dis)
generate_point_cloud_full(full_input, root, name, point_dis)
generate_surfel_cloud(full_rebuild_points, full_rebuild_normal, root, name, normal_angle)

