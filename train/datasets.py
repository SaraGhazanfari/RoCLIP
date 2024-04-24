import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class COCOFlickrDataset(Dataset):
    def __init__(
            self,
            model,
            image_processor,
            image_dir_path,
            annotations_path,
            transform=None,
            is_flickr=False,
            prefix=None,
    ):
        self.model = model
        self.image_processor = image_processor
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))["annotations"]
        self.is_flickr = is_flickr
        self.transform = transform
        self.prefix = prefix
        self.caption_list = self.model._prepare_text(
            [self.model.get_caption_prompt(ann["caption"]) for ann in self.annotations])

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"
        else:
            return f"{self.image_dir_path}/{self.prefix}{self.annotations[idx]['image_id']:012d}.jpg"

    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx))
        caption = self.caption_list[idx]  # self.annotations[idx]["caption"]
        image = self.image_processor([[image]]).half()

        # batch_text = []
        # batch_text.append(self.model.get_caption_prompt(caption))
        # print(batch_text)
        # caption = self.model._prepare_text(batch_text)
        print(image.shape)
        print(caption.shape)
        return image,


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        # target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return sample, target
