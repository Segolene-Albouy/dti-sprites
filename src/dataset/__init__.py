from .cosegmentation import WeizmannHorseDataset
from .gtsrb import GTSRB8Dataset
from .affnist import AffNISTTestDataset
from .hdf5 import FRGCDataset
from .coa import CoADataset
from .multi_object import DSpritesGrayDataset, TetrominoesDataset, CLEVR6Dataset
from .instagram import InstagramDataset
from .torchvision import (
    SVHNDataset,
    FashionMNISTDataset,
    MNISTDataset,
    MNISTTestDataset,
    MNISTColorDataset,
    MNIST1kDataset,
    USPSDataset,
)
from .raw import FleuronsDataset, LettersDataset


def get_dataset(dataset_name):
    return {
        # Cosegmentation
        "weizmann_horse": WeizmannHorseDataset,
        # Custom
        "affnist_test": AffNISTTestDataset,
        "gtsrb8": GTSRB8Dataset,
        "instagram": InstagramDataset,
        "frgc": FRGCDataset,
        # MultiObject
        "clevr6": CLEVR6Dataset,
        "dsprites_gray": DSpritesGrayDataset,
        "tetrominoes": TetrominoesDataset,
        # Torchvision
        "fashion_mnist": FashionMNISTDataset,
        "mnist": MNISTDataset,
        "mnist_test": MNISTTestDataset,
        "mnist_color": MNISTColorDataset,
        "mnist_1k": MNIST1kDataset,
        "svhn": SVHNDataset,
        "usps": USPSDataset,
        # Fleurons
        "fleurons": FleuronsDataset,
        # Letters
        "letters": LettersDataset,
        # CoA
        "coa": CoADataset,
    }[dataset_name]


def get_subset(dataset_name):
    if dataset_name == "fleurons":
        return FleuronsDataset
    else:
        NotImplementedError()
