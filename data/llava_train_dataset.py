import json
import os.path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CC3MDataset(Dataset):
    def __init__(self, img_root, annotations_path):
        self.full_annotation_list = json.load(open(annotations_path, "r"))
        self.img_root = img_root
        self.img_path_list = os.listdir(self.img_root)
        print(self.__len__())

    def __len__(self):
        return len(self.full_annotation_list)

    def get_img_path(self, img_name):
        return os.path.join(self.img_root, img_name)

    def __getitem__(self, idx):
        img_path = self.get_img_path(self.full_annotation_list[idx]['image'])
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "question": self.full_annotation_list[idx]['conversations'][0]['value'],
            "question_id": self.full_annotation_list[idx]['id'],
            "answers": self.full_annotation_list[idx]['conversations'][1]['value']
        }
        return results
