from functools import lru_cache
from PIL import Image
import json

from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, ToTensor, Resize

from ..utils import coerce_to_path_and_check_exist
from ..utils.image import IMG_EXTENSIONS
from ..utils.path import DATASETS_PATH


class CoADataset(TorchDataset):
    root = DATASETS_PATH
    name = "coa"
    n_channels = 3
    inp_exts = IMG_EXTENSIONS

    def __init__(self, split, subset, tag, **kwargs):
        self.data_path = (
            coerce_to_path_and_check_exist(self.root / self.name / tag) / split
        )
        self.split = split
        self.tag = tag

        json_file = coerce_to_path_and_check_exist(
            self.root / self.name / tag
        ) / kwargs.get("json_file")
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.n_classes = 5
        self.size = len(self.data)

        img_size = kwargs.get("img_size")

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.crop = True
        else:
            assert len(img_size) == 2
            self.img_size = img_size
            self.crop = False

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp = self.transform(
            Image.open(
                str(self.data_path) + "/" + self.data[idx]["filename"] + ".jpg"
            ).convert("RGB")
        )
        return inp, self.data[idx]["label"], []

    @property
    @lru_cache()
    def transform(self):
        if self.crop:
            size = self.img_size[0]
            transform = [Resize(size), CenterCrop(size), ToTensor()]
        else:
            transform = [Resize(self.img_size), ToTensor()]
        return Compose(transform)
