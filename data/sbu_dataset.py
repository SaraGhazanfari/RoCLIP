import os
from typing import Dict

from PIL import Image
from torchvision.datasets import SBU


class CompleteSBU(SBU):
    def __init__(self, root, transform):
        super(CompleteSBU, self).__init__(root=root, transform=transform, download=False)

    def __getitem__(self, index: int) -> Dict:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a caption for the photo.
        """
        filename = os.path.join(self.root, "dataset", self.photos[index])
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        target = self.captions[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            "image": img,
            "caption": target,
            "image_id": self.photos[index],
        }
