import os.path as osp

import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2 as cv


class IBUG(Dataset):
    def __init__(self, images_root_path, landmarks_folder_path, transform=None, test=False):
        self.images_root_path = images_root_path
        self.landmarks_folder_path = landmarks_folder_path
        self.test = test

        landmarks_file_name = 'train.txt'
        if self.test:
            landmarks_file_name = 'ibug.txt'

        self.landmarks_file = open(osp.join(landmarks_folder_path, landmarks_file_name), 'r')
        self.sample_info = self._read_samples_info()
        self.transform = transform

    def _read_samples_info(self):
        samples = []

        landmarks_file_lines = self.landmarks_file.readlines()

        for i in tqdm(range(len(landmarks_file_lines))):
            line = landmarks_file_lines[i].strip()

            name = line.split('/')[0]
            img_name = name + '.jpg'
            img_path = osp.join(self.images_root_path, img_name)

            landmark_name = name + '.json'
            landmarks_path = osp.join(self.landmarks_folder_path, landmark_name)
            samples.append((img_path, landmarks_path))


        return samples
    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        # img = cv.cvtColor(cv.imread(self.sample_info[idx][0], cv.IMREAD_GRAYSCALE), cv.COLOR_GRAY2RGB)
        img = cv.imread(self.sample_info[idx][0], cv.IMREAD_COLOR)
        with open(self.sample_info[idx][1], 'r') as f:
            landmarks = np.array(json.load(f))
        data = {'img': img, 'landmarks': landmarks}
        if self.transform:
            data = self.transform(data)
        return data
