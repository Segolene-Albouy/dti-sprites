from abc import ABCMeta
from functools import lru_cache
from PIL import Image

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import ToTensor, Compose, Resize

from ..utils import coerce_to_path_and_check_exist, use_seed
from ..utils.path import DATASETS_PATH
import os


class _AbstractMultiObjectDataset(TorchDataset):
    __metaclass__ = ABCMeta
    root = DATASETS_PATH
    name = NotImplementedError
    n_channels = 3
    n_classes = NotImplementedError
    img_size = NotImplementedError
    N = NotImplementedError
    instance_eval = True

    def __init__(self, split, **kwargs):
        self.data_path = coerce_to_path_and_check_exist(self.root / self.name)
        self.split = split
        self.eval_mode = kwargs.get("eval_mode", False) or split == "test"
        self.eval_semantic = kwargs.get("eval_semantic", False)
        self.eval_qualitative = kwargs.get("eval_qualitative", False)

        if self.eval_mode:
            self.size = 320
        elif split == "val":
            with use_seed(42):
                self.val_indices = np.random.choice(range(self.N), 100, replace=False)
            self.size = 100
        else:
            self.size = self.N

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        path = self.data_path
        if self.split == "val":
            idx = self.val_indices[idx]
        inp = self.transform(Image.open(path / "images" / f"{idx}.png").convert("RGB"))
        if self.eval_semantic:
            label = (
                self.transform_gt(
                    Image.open(path / "sem_masks" / f"{idx}.png").convert("L")
                )
                * 255
            ).long()
        else:
            if os.path.exists(path / "masks"):
                label = (
                    self.transform_gt(
                        Image.open(path / "masks" / f"{idx}.png").convert("L")
                    )
                    * 255
                ).long()
            else:
                label = -1
        return inp, label, [], str(self.data_path / "images" / f"{idx}.png")

    @property
    @lru_cache()
    def transform(self):
        return Compose([ToTensor()])

    @property
    @lru_cache()
    def transform_gt(self):
        return Compose([ToTensor()])


class DSpritesGrayDataset(_AbstractMultiObjectDataset):
    name = "dsprites_gray"
    img_size = (64, 64)
    N = 60000
    n_classes = 4


class CLEVR6Dataset(_AbstractMultiObjectDataset):
    name = "clevr6"
    img_size = (128, 128)
    N = 34963
    n_classes = 7

class CLEVRDataset(_AbstractMultiObjectDataset):
    name = "clevr"
    img_size = (128, 128)
    N = 100000
    n_classes = 7

class TetrominoesDataset(_AbstractMultiObjectDataset):
    name = "tetrominoes"
    img_size = (35, 35)
    N = 60000
    n_classes = 20


class FleuronsCompDataset(_AbstractMultiObjectDataset):
    name = "fleuron_compounds"
    img_size = (64, 64)
    N = 100
    n_classes = 72
    pred_class = True

    def __init__(self, split, **kwargs):
        super().__init__(split, **kwargs)
        self.data_path = coerce_to_path_and_check_exist(self.root / self.name)
        self.input_files = os.listdir(str(self.data_path) + "/images_512")
        self.seg_eval = kwargs.get("seg_eval", True)
        self.instance_eval = kwargs.get("instance_eval", True)
        if self.split == "val":
            raise ValueError

    def __getitem__(self, idx):
        inp = self.transform(
            Image.open(self.data_path / "images_512" / self.input_files[idx]).convert(
                "RGB"
            )
        )
        return inp, -1, [], str(self.input_files[idx])

    @property
    @lru_cache()
    def transform(self):
        return Compose([Resize(self.img_size), ToTensor()])


class FleuronsCompSyntDataset(_AbstractMultiObjectDataset):
    name = "fleuron_compounds_synt"
    img_size = (128, 128)
    N = 50436
    n_classes = 72
    pred_class = True

    def __init__(self, split, **kwargs):
        super().__init__(split, **kwargs)
        self.data_path = coerce_to_path_and_check_exist(self.root / self.name)
        self.input_files = os.listdir(str(self.data_path) + "/img")
        self.seg_eval = kwargs.get("seg_eval", True)
        self.instance_eval = kwargs.get("instance_eval", True)
        self.tr = self.transform()
        if self.split == "val":
            self.size = 0

    def __getitem__(self, idx):
        inp = self.tr(
            Image.open(self.data_path / "img" / self.input_files[idx]).convert("RGB")
        )
        return inp, -1, [], str(self.input_files[idx])

    @property
    @lru_cache()
    def transform(self):
        return Compose([Resize(self.img_size), ToTensor()])
