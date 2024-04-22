import json
import os

from torchvision.datasets import SBU

id_list = []
with open('/data/sg7457/coco/captions_val2014.json', 'r') as fp:
    data = json.load(fp)

for i in range(len(data['images'])):
    id_list.append(data['images'][i]['id'])

with open('/data/sg7457/coco/captions_train2014.json', 'r') as fp:
    train_data = json.load(fp)

for i in range(len(train_data['images'])):
    id_list.append(train_data['images'][i]['id'])

print(len(set(id_list)))


dataset = SBU(root='/data/sg7457', transform=None, download=False)
data_dict = {"images": []}
url_list = []
with open(os.path.join(dataset.root, "dataset", "SBU_captioned_photo_dataset_urls.txt")) as fh:
    for line in fh:
        url_list.append(line.rstrip())

for idx in range(len(dataset)):
    file_name = dataset.photos[idx]
    data_dict["images"].append({'coco_url': url_list[idx], 'file_name': file_name, 'id': id_list[idx]})

with open('dataset/ann.json', 'w') as fp:
    json.dump(data_dict, fp)
