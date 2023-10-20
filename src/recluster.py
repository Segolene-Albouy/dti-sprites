import json
import argparse
import os
import shutil
import time
import yaml
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import get_subset, get_dataset
from .model import get_model
from .model.tools import count_parameters, safe_model_state_dict
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import use_seed, coerce_to_path_and_check_exist, coerce_to_path_and_create_dir
from .utils.image import convert_to_img, save_gif
from .utils.logger import get_logger, print_info, print_warning
from .utils.metrics import AverageTensorMeter, AverageMeter, Metrics, Scores
from .utils.path import CONFIGS_PATH, RUNS_PATH
from .trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to train a NN model specified by a YML config")
    parser.add_argument("-t", "--tag", nargs="?", type=str, required=True, help="Run tag of the experiment")
    parser.add_argument("-c", "--config", nargs="?", type=str, required=True, help="Config file name")
    args = parser.parse_args()

    assert args.tag is not None and args.config is not None
    config = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    seed = cfg["training"].get("seed", 4321)
    dataset = cfg["dataset"]["name"]

    n_levels = cfg["training"].get("n_dup", 0)
    sets = ['0','00','01','000','001','010','011','0000','0001','0010','0011','0100','0101','0110','0111','00000','00001','00010','00011','00100','00101','00110','00111','01000','01001','01010','01011','01100','01101','01110','01111','000000','000001','000010','000011','000100','000101','000110','000111','001000','001001','001010','001011','001100','001101','001110','001111','010000','010001','010010','010011','010100','010101','010110','010111','011000','011001','011010','011011','011100','011101','011110','011111'] 

    old_id = {}
    old_distance = {}
    sets = np.array(sets)
    sid = ['000000','000001','000010','000011','000100', '000101','000110','000111','001000','001001','001010','001011','001100','001101','001110','001111','010000','010001','010010','010011','010100','010101','010110',    '010111','011000','011001','011010','011011','011100','011101','011110','011111'] 
    for leaf in sid:
        run_dir = RUNS_PATH / dataset / args.tag
        run_dir = str(run_dir) + '_' + leaf
        leaf_id = np.load(run_dir + '/leaf_id_by_sample.npy')
        path_ = np.load(run_dir + '/paths.npy', allow_pickle=True)
        for idx in range(len(leaf_id)):
            old_id[path_[idx]] = leaf_id[idx]
        rec_err = np.load(run_dir + '/rec_err_by_sample.npy')
        for idx in range(len(rec_err)):
            old_distance[path_[idx]] = float(rec_err[idx])
    
    with open("old_dist_by_sample.json", "w") as fp:
        json.dump(old_distance, fp)
    with open("old_id_by_sample.json", "w") as fp:
        json.dump(old_id, fp)
    
    for set_ in sid: # sets:
        run_dir = RUNS_PATH / dataset / args.tag
        run_dir = str(run_dir) + '_' + set_
        trainer = Trainer(config, run_dir, seed=seed, recluster=True)
        new_dists = trainer.run(seed=seed, recluster=True)
        for sample, new_dist, new_id in new_dists:
            new_dist = float(new_dist)
            if new_dist+1e-5 <= old_distance[sample]:
                old_distance[sample] = new_dist

                old_id[sample] = set_+str(new_id)
    
    with open("new_dist_by_sample_leaf.json", "w") as fp:
        json.dump(old_distance, fp)
    with open("new_id_by_sample_leaf.json", "w") as fp:
        json.dump(old_id, fp)
