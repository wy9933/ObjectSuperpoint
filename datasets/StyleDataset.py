import os
import torch
import numpy as np
import torch.utils.data as data
from utils.utils import *


class StyleDataset(data.Dataset):
    def __init__(self, args, config, split='train'):
        self.split = split

        assert self.split in ['train', 'test', 'val', 'whole']

        self.data_root = config.dataset.DATA_ROOT
        self.data_dir = config.dataset.DATA_DIR
        self.split_dir = config.dataset.SPLIT_DIR
        self.npoints = config.dataset.N_POINTS

        self.permutation = np.arange(self.npoints)

        if split == 'train' or split == 'whole':
            self.data_list_file = os.path.join(self.data_root, self.split_dir, 'train.txt')
        else:
            self.data_list_file = os.path.join(self.data_root, self.split_dir, 'test.txt')

        self.sample_points_num = config.dataset[self.split].npoints

        print_log(args, f'Sample out {self.sample_points_num} points')
        print_log(args, f'Open file {self.data_list_file}')

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        if split == 'whole':
            test_data_list_file = os.path.join(self.data_root, self.split_dir, 'test.txt')
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(args, f'Open file {test_data_list_file}')
            lines = lines + test_lines

        self.file_list = []
        for line in lines:
            line = line.strip()
            class_name = line.split('_')[0]
            model_id = line.split('_')[1].split('.')[0]
            self.file_list.append({
                'class_name': class_name,
                'model_id': model_id,
                'file_path': line
            })
        print_log(args, f'{len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = np.load(os.path.join(self.data_root, self.data_dir, sample['file_path'])).astype(np.float32)
        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample['class_name'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)