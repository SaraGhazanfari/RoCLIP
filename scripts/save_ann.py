import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import SBU


def save_ann_file_for_sbu():
    dataset = SBU(root='/data/sg7457', transform=None, download=False)
    data_dict = {"images": [], "annotations": []}
    url_list = []
    with open(os.path.join(dataset.root, "dataset", "SBU_captioned_photo_dataset_urls.txt")) as fh:
        for line in fh:
            url_list.append(line.rstrip())

    for idx in range(len(dataset)):
        file_name = dataset.photos[idx]
        data_dict["images"].append(
            {'coco_url': url_list[idx], 'file_name': file_name, 'id': file_name})
        data_dict["annotations"].append(
            {'caption': dataset.captions[idx], 'id': file_name, 'image_id': file_name})

    with open('dataset/ann.json', 'w') as fp:
        json.dump(data_dict, fp)


class CC3MCaptionDataset(Dataset):
    def __init__(self, img_root, annotations_path):
        self.full_annotation_list = json.load(open(annotations_path, "r"))
        self.img_root = img_root
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
            "image_id": self.full_annotation_list[idx]['id'],
            "caption": [self.full_annotation_list[idx]['conversations'][1]['value']]
        }
        return results


def save_ann_file_for_cc3m():
    dataset = CC3MCaptionDataset(img_root='/data/sg7457/CC-3M', annotations_path='/data/sg7457/CC-3M/chat.json')
    data_dict = {"images": [], "annotations": []}

    for row in dataset:
        data_dict["images"].append(
            {'coco_url': row['image_id'], 'file_name': row['image_id'], 'id': row['image_id']})
        data_dict["annotations"].append(
            {'caption': row['caption'], 'id': row['image_id'], 'image_id': row['image_id']})

    with open('/data/sg7457/CC-3M/ann.json', 'w') as fp:
        json.dump(data_dict, fp)


if __name__ == '__main__':
    save_ann_file_for_cc3m()
