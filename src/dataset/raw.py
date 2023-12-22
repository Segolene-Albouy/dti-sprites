from abc import ABCMeta
from functools import lru_cache
from PIL import Image

from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from ..utils import coerce_to_path_and_check_exist, get_files_from_dir
from ..utils.image import IMG_EXTENSIONS
from ..utils.path import DATASETS_PATH
from pathlib import Path
import numpy as np
import torch


class _AbstractCollectionDataset(TorchDataset):
    """Abstract torch dataset from raw files collections associated to tags."""

    __metaclass__ = ABCMeta
    root = DATASETS_PATH
    name = NotImplementedError
    n_channels = 3
    include_recursive = False
    output_paths = False

    def __init__(self, split, subset, img_size, **kwargs):
        tag = kwargs.get("tag", "")
        self.data_path = (
            coerce_to_path_and_check_exist(self.root / self.name / tag) / split
        )
        self.split = split
        if not isinstance(subset, type(None)):
            input_files = [Path(p) for p in sorted(subset)]
        else:
            try:
                input_files = get_files_from_dir(
                    self.data_path,
                    IMG_EXTENSIONS,
                    sort=True,
                    recursive=self.include_recursive,
                )
                input_files = [p for p in input_files if not "/__" in str(p)]
            except FileNotFoundError:
                input_files = []

        self.input_files = input_files
        self.labels = [-1] * len(input_files)
        self.n_classes = 0
        self.size = len(self.input_files)

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.crop = True
        else:
            assert len(img_size) == 2
            self.img_size = img_size
            self.crop = False

        # if self.size > 0:
        #    sample_size = Image.open(self.input_files[0]).size
        #    if min(self.img_size) > min(sample_size):
        #        raise ValueError("img_size too big compared to a sampled image size, adjust it or upscale dataset")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.open(self.input_files[idx])
        if img.mode == "RGBA":
            alpha = img.split()[-1]
        else:
            h, w = img.size
            alpha = Image.new("L", (h, w), (255))
        inp = self.transform(img.convert("RGB"))
        alpha = self.transform(alpha)
        if self.output_paths:
            return inp, self.labels[idx], alpha, str(self.input_files[idx])
        return inp, self.labels[idx], alpha  # str(self.input_files[idx])

    @property
    @lru_cache()
    def transform(self):
        if self.crop:
            size = self.img_size[0]
            transform = [Resize(size), CenterCrop(size), ToTensor()]
        else:
            transform = [Resize(self.img_size), ToTensor()]
        return Compose(transform)


class MegaDepthDataset(_AbstractCollectionDataset):
    name = "megadepth"


class GenericDataset(_AbstractCollectionDataset):
    name = "generic"
    include_recursive = True


class LettersDataset(_AbstractCollectionDataset):
    name = "Lettre_e"
