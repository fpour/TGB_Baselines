import torch
from torch_geometric.data import TemporalData
import random
import os
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from typing import Optional, Dict, Any, Tuple
import math
import math

def set_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'INFO: fixed random seed: {random_seed}')


def list2csv(lst: list,
             fname: str,
             delimiter: str = ",",
             fmt: str = '%i'):
    out_list = np.array(lst)
    np.savetxt(fname, out_list, delimiter=delimiter,  fmt=fmt)



def remove_duplicate_edges(data):

    src = data.src.cpu().numpy()
    dst = data.dst.cpu().numpy()
    ts = data.t.cpu().numpy()
    msg = data.msg.cpu().numpy()
    y = data.y.cpu().numpy()

    query = np.stack([src, dst, ts], axis=0)
    uniq, idx = np.unique(query, axis=1, return_index=True)
    print ("number of edges reduced from ", query.shape[1], " to ", uniq.shape[1])

    src = torch.from_numpy(src[idx])
    dst = torch.from_numpy(dst[idx])
    ts = torch.from_numpy(ts[idx])
    msg = torch.from_numpy(msg[idx])
    y = torch.from_numpy(y[idx])

    new_data = TemporalData(
            src=src,
            dst=dst,
            t=ts,
            msg=msg,
            y=y,
        )
    return new_data


def get_snapshot_batches(timestamps):
    r"""
    construct batches of edges based on timestamps
    #! assume timestamp always start from 0, is sorted and no gap
    Parameters:
        timestamps: timestamps array
    Returns:
        index_dict: a dictionary of start and end indexes for each snapshot
    """
    index_dict = {}
    values, indices = np.unique(timestamps, return_index=True)
    for i in range(len(values)):
        if (i == len(values) - 1):
            index_dict[values[i]] = (indices[i], len(timestamps))
        else:
            index_dict[values[i]] = (indices[i], indices[i+1])
    return index_dict



def generate_splits(
        full_data: Dict[str, Any],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        r"""Generates train, validation, and test splits from the full dataset
        Args:
            full_data: dictionary containing the full dataset
            val_ratio: ratio of validation data
            test_ratio: ratio of test data
        Returns:
            train_data: dictionary containing the training dataset
            val_data: dictionary containing the validation dataset
            test_data: dictionary containing the test dataset
        """
        #! split by snapshot id instead of edge numbers
        ts_list = np.unique(full_data["timestamps"]).tolist()
        ts_list = sorted(ts_list)

        val_time, test_time = list(
            np.quantile(
                ts_list,
                [(1 - val_ratio - test_ratio), (1 - test_ratio)],
            )
        )

        #! changes added to ensure it works with integer correctly
        val_time = math.ceil(val_time)
        test_time = math.ceil(test_time)

        timestamps = full_data["timestamps"]

        train_mask = timestamps <= val_time
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        return train_mask, val_mask, test_mask

def convert2Torch(src,
                  dst,
                  ts):
    """
    convert numpy array to torch tensor
    Parameters:
        src: numpy array
        dst: numpy array
        ts: numpy array
    """
    src = torch.from_numpy(src)
    dst = torch.from_numpy(dst)
    ts = torch.from_numpy(ts)
    if src.dtype != torch.int64:
        src = src.long()

    # destination tensor must be of type int64
    if dst.dtype != torch.int64:
        dst = dst.long()

    # timestamp tensor must be of type int64
    if ts.dtype != torch.int64:
        ts = ts.long()
    return src, dst, ts

def convert_to_torch_extended(src, dst, ts, lbl, edge_idx):
    """
    convert numpy array to torch tensor
    NOTE: extended version is required, since NAT gets labels and edge_indices as well

    Parameters:
        src, dst, ts, lbl, edge_idx: numpy array
    """
    # convert to Torch tensors
    src = torch.from_numpy(src)
    dst = torch.from_numpy(dst)
    ts = torch.from_numpy(ts)
    lbl = torch.from_numpy(lbl)
    edge_idx = torch.from_numpy(edge_idx)

    # type checks
    if src.dtype != torch.int64:
        src = src.long()

    if dst.dtype != torch.int64:
        dst = dst.long()

    if ts.dtype != torch.int64:
        ts = ts.long()

    if lbl.dtype != torch.int64:
        lbl = lbl.long()

    if edge_idx.dtype != torch.int64:
        edge_idx = edge_idx.long()

    return src, dst, ts, lbl, edge_idx

def mkdirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder