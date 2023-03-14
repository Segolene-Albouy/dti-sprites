from .cosegmentation import WeizmannHorseDataset
from .gtsrb import GTSRB8Dataset
from .multi_object import DSpritesGrayDataset, TetrominoesDataset, CLEVR6Dataset
from .instagram import InstagramDataset
from .torchvision import SVHNDataset
from .raw import FleuronsDataset, LettersDataset

def get_dataset(dataset_name):
    return {
        # Cosegmentation
        'weizmann_horse': WeizmannHorseDataset,

        # Custom
        'gtsrb8': GTSRB8Dataset,
        'instagram': InstagramDataset,

        # MultiObject
        'clevr6': CLEVR6Dataset,
        'dsprites_gray': DSpritesGrayDataset,
        'tetrominoes': TetrominoesDataset,

        # Torchvision
        'svhn': SVHNDataset,
        
        # Fleurons
        'fleurons': FleuronsDataset,
        
        # Letters
        'letters': LettersDataset,
    }[dataset_name]

def get_subset(dataset_name):
    if dataset_name=='fleurons':
        return FleuronsDataset
    else:
        NotImplementedError()
