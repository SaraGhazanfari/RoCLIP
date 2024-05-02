import copy
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from llava.constants import IGNORE_INDEX


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

        attention_mask, input_ids, labels = self._process_text(caption)
        attention_mask, input_ids, labels = self._pad_text(attention_mask, input_ids, labels)
        return image, input_ids, labels, attention_mask

    def _pad_text(self, attention_mask, input_ids, labels):
        max_length = 100
        input_ids = input_ids[0, :max_length]
        labels = labels[0, :max_length]
        attention_mask = attention_mask[0, :max_length]
        if input_ids.shape[0] < max_length:
            pad_token_tensor = torch.tensor(
                [self.model.config.pad_token_id] * (max_length - input_ids.shape[0]))
            attention_mask_tensor = torch.tensor([False] * (max_length - input_ids.shape[0]))
            labels = torch.cat((labels, pad_token_tensor), dim=0)
            input_ids = torch.cat((input_ids, pad_token_tensor), dim=0)
            attention_mask = torch.cat((attention_mask, attention_mask_tensor), dim=0)
        return attention_mask, input_ids, labels

    def _process_text(self, caption):
        batch_text = []
        batch_text.append(self.model.get_caption_prompt(caption))
        input_ids = self.model._prepare_text(batch_text, device='cpu')
        context_only = batch_text[0].get_prompt().split("ASSISTANT:")[0] + "ASSISTANT:"
        context_len = len(self.model.tokenizer.encode(context_only))
        labels = copy.deepcopy(input_ids)
        labels[:, :context_len] = IGNORE_INDEX
        attention_mask = input_ids.ne(self.model.tokenizer.pad_token_id)
        return attention_mask, input_ids, labels


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        # target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return sample, target
