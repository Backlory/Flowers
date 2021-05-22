import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class PSR_Dataset(Dataset):
    def __init__(self, dataset_path):
        self.data_paths_list = []
        self.data_labels_list = []
        self.num2label = {  0:'paper',
                            1:'scissors',
                            2:'rock'
                            }
        self.dataset_paths = [os.path.join(dataset_path, label) for label in ['paper','scissors','rock']]
        for idx, dataset_path in enumerate(self.dataset_paths):
            data_path_list = self.get_read_file_list(dataset_path)
            self.data_paths_list = self.data_paths_list + data_path_list
            self.data_labels_list = self.data_labels_list + [idx]*len(data_path_list)
    #
    def __getitem__(self, index):
        data_path = self.data_paths_list[index]
        data_label = self.data_labels_list[index]
        img = cv2.imread(data_path, 1)
        return img, data_label
    #
    def __len__(self):
        return len(self.data_labels_list)
    #
    def get_read_file_list(self, path):
        """eg.
            >>read_file_list('data\\rock\\')
            ['data\\rock\\rock07-k03-118.png',...]
        """
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        return file_path_list
