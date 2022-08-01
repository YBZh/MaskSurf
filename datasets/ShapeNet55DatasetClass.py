import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
from utils import misc
import pickle

class_dict = {"02691156": "airplane", "02747177": "trash bin", "02773838": "bag", "02801938": "basket", "02808440": "bathtub", "02818832": "bed", "02828884": "bench", "02843684": "birdhouse", "02871439": "bookshelf", "02876657": "bottle", "02880940": "bowl", "02924116": "bus", "02933112": "cabinet", "02942699": "camera", "02946921": "can", "02954340": "cap", "02958343": "car", "02992529": "cellphone", "03001627": "chair", "03046257": "clock", "03085013": "keyboard", "03207941": "dishwasher", "03211117": "display", "03261776": "earphone", "03325088": "faucet", "03337140": "file cabinet", "03467517": "guitar", "03513137": "helmet", "03593526": "jar", "03624134": "knife", "03636649": "lamp", "03642806": "laptop", "03691459": "loudspeaker", "03710193": "mailbox", "03759954": "microphone", "03761084": "microwaves", "03790512": "motorbike", "03797390": "mug", "03928116": "piano", "03938244": "pillow", "03948459": "pistol", "03991062": "flowerpot", "04004475": "printer", "04074963": "remote", "04090263": "rifle", "04099429": "rocket", "04225987": "skateboard", "04256520": "sofa", "04330267": "stove", "04379243": "table", "04401088": "telephone", "04460130": "tower", "04468005": "train", "04530566": "watercraft", "04554684": "washer"}

taxonomy_list = list(class_dict.keys())
for label_index in range(len(taxonomy_list)):
    class_dict[taxonomy_list[label_index]] = label_index

def map_taxonomy_to_label(taxonomy):
    return class_dict[taxonomy]


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
class ShapeNetClass(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.process_data = True
        # self.npoints = config.N_POINTS
        self.sample_points_num = config.N_POINTS
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        # test_data_list_file = os.path.join(self.data_root, 'test.txt')
        

        # self.whole = config.get('whole')
        # print('NOTE!!!!! the whole in the shapenet dataset is:', self.whole)  # None

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        # if self.whole:
        #     with open(test_data_list_file, 'r') as f:
        #         test_lines = f.readlines()
        #     print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
        #     lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')


        assert (self.subset == 'train' or self.subset == 'test')
        # shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
        #                  in range(len(shape_ids[split]))]
        self.save_path = os.path.join(self.data_root, 'shapenet_%s_%dpts.dat' % (self.subset, 8192))
        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path, logger = 'ShapeNet')
                self.list_of_points = [None] * len(self.file_list)
                self.list_of_labels = [None] * len(self.file_list)

                for idx in range(len(self.file_list)):
                    print('%d/%d' % (idx, len(self.file_list)))
                    sample = self.file_list[idx]
                    data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
                    data = data[:, :3].copy()
                    # data = farthest_point_sample(data, self.sample_points_num)  # N 3
                    # data = self.pc_norm(data)
                    # data = torch.from_numpy(data).float()
                    label = map_taxonomy_to_label(sample['taxonomy_id'])

                    self.list_of_points[idx] = data
                    self.list_of_labels[idx] = label

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log('Load processed data from %s...' % self.save_path, logger = 'ShapeNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        # self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    # def random_sample(self, pc, num):
    #     np.random.shuffle(self.permutation)
    #     pc = pc[self.permutation[:num]]
    #     return pc
        
    def __getitem__(self, idx):
        if self.process_data:
            data, label = self.list_of_points[idx], self.list_of_labels[idx]
        else:
            sample = self.file_list[idx]
            data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
            data = data[:, :3].copy()
            label = map_taxonomy_to_label(sample['taxonomy_id'])

        data = self.pc_norm(data)
        pt_idxs = np.arange(0, data.shape[0])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        data = data[pt_idxs].copy()
        data = torch.from_numpy(data).float()

        return 'ScanObjectNN', 'sample', (data, label)
        # return sample['taxonomy_id'], sample['model_id'], data


    def __len__(self):
        return len(self.file_list)