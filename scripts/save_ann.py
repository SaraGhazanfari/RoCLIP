import json
import os

from torchvision.datasets import SBU

dataset = SBU(root='/data/sg7457', transform=None, download=False)
data_dict = {"images": []}
url_list = []
with open(os.path.join(dataset.root, "dataset", "SBU_captioned_photo_dataset_urls.txt")) as fh:
    for line in fh:
        url_list.append(line.rstrip())

for idx in range(len(dataset)):
    file_name = dataset.photos[idx]
    data_dict["images"].append({'coco_url': url_list[idx], 'file_name': file_name, 'id': idx})

with open('dataset/ann.json', 'w') as fp:
    json.dump(data_dict, fp)
