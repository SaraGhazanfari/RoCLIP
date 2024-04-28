import json

import torch
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

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"
        else:
            return f"{self.image_dir_path}/{self.prefix}{self.annotations[idx]['image_id']:012d}.jpg"

    def __getitem__(self, idx):
        #
        image = Image.open(self.get_img_path(idx))
        caption = self.annotations[idx]["caption"]
        image = self.image_processor([[image]]).half().squeeze(0)

        batch_text = []
        batch_text.append(self.model.get_caption_prompt(caption))
        input_ids, labels, attention_mask, past_key_values = self.model._prepare_text(batch_text, past_key_values=None,
                                                                                      to_device=True)
        prompt_max_length = 100
        label_max_length = 20
        input_ids = input_ids[0, :prompt_max_length]
        labels = labels[0, :label_max_length]
        attention_mask = attention_mask[0, :prompt_max_length]
        if input_ids.shape[0] < prompt_max_length:
            pad_token_tensor = torch.tensor(
                [self.model.config.pad_token_id] * (prompt_max_length - input_ids.shape[0]))
            attention_mask_tensor = torch.tensor([False] * (prompt_max_length - input_ids.shape[0]))
            input_ids = torch.cat((input_ids, pad_token_tensor), dim=0)
            attention_mask = torch.cat((attention_mask, attention_mask_tensor), dim=0)

        if labels.shape[0] < label_max_length:
            pad_token_tensor = torch.tensor(
                [self.model.config.pad_token_id] * (label_max_length - input_ids.shape[0]))
            labels = torch.cat((labels, pad_token_tensor), dim=0)
        return image, input_ids, labels, attention_mask


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        # target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return sample, target
