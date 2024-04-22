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

    def __len__(self):
        return len(self.full_annotation_list)

    def get_img_path(self, img_name):
        return os.path.join(self.img_root, img_name)

    def get_from_id(self, question_id):
        count = 0
        for path in self.img_path_list:
            if str(int(question_id)) in path:
                image_path = path
                count += 1
        print(count)
        print(question_id)
        return image_path

    def __getitem__(self, idx):
        img_path = self.get_img_path(self.get_from_id(self.full_annotation_list[idx]['id']))
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "question": self.full_annotation_list[idx]['conversations'][0]['value'],
            "question_id": self.full_annotation_list[idx]['id'],
            "answers": self.full_annotation_list[idx]['conversations'][1]['value']
        }
        return results
