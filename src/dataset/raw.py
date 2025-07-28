from abc import ABCMeta
from functools import lru_cache
from PIL import Image

from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from ..utils import coerce_to_path_and_check_exist, get_files_from_dir
from ..utils.image import IMG_EXTENSIONS
from ..utils.path import DATASETS_PATH
from pathlib import Path


class _AbstractCollectionDataset(TorchDataset):
    """Abstract torch dataset from raw files collections associated to tags."""

    __metaclass__ = ABCMeta
    root = DATASETS_PATH
    name = NotImplementedError
    n_channels = 3
    include_recursive = False

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
        if self.cache_images and idx in self._image_cache:
            print(f"🏞️ Using cached image for index {idx}!")
            return self._image_cache[idx]

        img = Image.open(self.input_files[idx])
        alpha = img.split()[-1] if img.mode == "RGBA" else Image.new("L", img.size, (255))

        # keep grayscale images as is
        inp = self.transform(img.convert("RGB") if self.n_channels >= 3 else img)
        alpha = self.transform(alpha)

        item = (inp, self.labels[idx], alpha, str(self.input_files[idx]))
        if self.cache_images:
            self._image_cache[idx] = item
        return item

    @property
    @lru_cache()
    def transform(self):
        if self.crop:
            size = self.img_size[0]
            transform = [Resize(size), CenterCrop(size), ToTensor()]
        else:
            transform = [Resize(self.img_size), ToTensor()]
        return Compose(transform)

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
