import gc
from abc import ABCMeta
from functools import lru_cache
from PIL import Image
import torch

from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from ..utils import coerce_to_path_and_check_exist, get_files_from_dir
from ..utils.image import IMG_EXTENSIONS, invert_tsf
from ..utils.path import DATASETS_PATH
from pathlib import Path


class _AbstractCollectionDataset(TorchDataset):
    """Abstract torch dataset from raw files collections associated to tags."""

    __metaclass__ = ABCMeta
    root = DATASETS_PATH
    name = NotImplementedError
    include_recursive = False
    output_paths = False

    def __init__(self, split, img_size, cache_images=True, **kwargs):
        tag = kwargs.get("tag", "")
        self.data_path = (
            coerce_to_path_and_check_exist(self.root / self.name / tag) / split
        )
        self.split = split

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
        self.n_channels = self.get_n_channels()
        self._max_dim = (0, 0) # width, height (PIL size)
        self._max_dim_ok = False

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.crop = True
        else:
            assert len(img_size) == 2
            self.img_size = (int(img_size[0]), int(img_size[1]))
            self.crop = False

        self.cache_images = cache_images
        self._image_cache = {}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_name = self.input_files[idx]
        if self.cache_images and idx in self._image_cache:
            print(f"🏞️ Using cached image {img_name}!")
            return self._image_cache[idx]

        img = Image.open(img_name)
        self._max_dim = (max(self._max_dim[0], img.size[0]), max(self._max_dim[1], img.size[1]))
        alpha = img.split()[-1] if img.mode == "RGBA" else Image.new("L", img.size, (255))

        # keep grayscale images as is
        inp = self.transform(img.convert("RGB") if self.n_channels >= 3 else img)
        alpha = self.transform(alpha)

        item = (inp, self.labels[idx], alpha, str(img_name))
        if self.cache_images:
            self._image_cache[idx] = item
        return item

    @property
    def max_dim(self):
        if len(self._image_cache) == self.size or self._max_dim_ok:
            return self._max_dim
        for i in range(self.size):
            img = Image.open(self.input_files[i])
            self._max_dim = (max(self._max_dim[0], img.size[0]), max(self._max_dim[1], img.size[1]))
        self._max_dim_ok = True
        return self._max_dim

    def clear_cache(self):
        del self._image_cache
        self._image_cache = {}
        self.cache_images = False
        gc.collect()
        print("🗑️ Image cache cleared!")

    def get_original(self, idx):
        img = Image.open(self.input_files[idx])
        # keep grayscale images as is
        return img.convert("RGB") if self.n_channels >= 3 else img


    @property
    @lru_cache()
    def transform(self):
        if self.crop:
            size = self.img_size[0]
            transform = [Resize(size), CenterCrop(size), ToTensor()]
        else:
            transform = [Resize(self.img_size), ToTensor()]
        return Compose(transform)

    def inverse_transform(self, idx):
        orig_size = self.get_original(idx).size
        return Compose([Resize(orig_size)])

    def get_resize_tsf(self, idx, inverse=False):
        """
        inverse: from model size to original size
        """
        orig_w, orig_h = self.get_original(idx).size  # (W, H)
        target_h, target_w = self.img_size  # (H, W)

        tsf = torch.tensor([
            [(target_w / orig_w), 0.0, 0.0],
            [0.0, (target_h / orig_h), 0.0]
        ], dtype=torch.float32)
        return invert_tsf(tsf) if inverse else tsf


    def get_n_channels(self):
        # try:
        #     if self.size == 0:
        #         return 3
        #
        #     img = Image.open(self.input_files[0])
        #     if img.mode == 'L':  # Grayscale
        #         return 1
        #     elif img.mode in ['RGB', 'YCbCr']:  # RGB-like
        #         return 3
        #     elif img.mode in ['RGBA', 'CMYK']:  # RGBA-like
        #         # return 4
        #         return 3 # alpha channel is handled separately
        # except Exception as e:
        #     print(f"Warning: Could not determine channels from first image: {e}")
        return 3


class MegaDepthDataset(_AbstractCollectionDataset):
    name = "megadepth"


class GenericDataset(_AbstractCollectionDataset):
    name = "generic"
    include_recursive = True


class LettersDataset(_AbstractCollectionDataset):
    name = "Lettre_e"


class CoADataset(_AbstractCollectionDataset):
    name = "coa_marion"
