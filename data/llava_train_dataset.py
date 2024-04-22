import json
import os.path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CC3MDataset(Dataset):
    def __init__(self, img_root, annotations_path):
        self.full_annotation_list = json.load(open(annotations_path, "r"))
        self.img_root = img_root

    def __len__(self):
        return len(self.full_annotation_list)

    def get_img_path(self, img_name):
        return os.path.join(self.img_root, img_name)

    def get_from_id(self, question_id):
        for row in self.full_annotation_list:
            if row['id'] == question_id:
                image_path = self.get_img_path(img_name=row['image'])
        image = torch.load(image_path)
        return image

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
