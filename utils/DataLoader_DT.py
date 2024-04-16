"""
Load and process "Discrete Time" data so that it can be fed to the NAT model
"""
from torch_geometric.data import TemporalData
from utils.data_util import load_dtdg
from utils.utils_func import generate_splits, convert_to_torch_extended, set_random, get_snapshot_batches
import timeit
import argparse
import os
import os.path as osp
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


def get_lpp_data_DT(dataset_name: str, time_scale: str = None,
                    val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    generate tgb data for link prediction task
    :param dataset_name (str): dataset name
    :param time_scale (str): if decided to discretize time, this specifies the time granularity
    :param val_ratio (float): the validation ratio
    :param test_ratio (float): the testing ratio
    """

    # =========================================================================
    # Data Loading
    dtdg, ts_list = load_dtdg(dataset_name, time_scale)

    full_data = dtdg.export_full_data()
    src_node_ids = full_data["sources"]
    dst_node_ids = full_data["destinations"]
    node_interact_times = full_data["timestamps"]
    labels = np.ones(len(src_node_ids))  # dummy variables
    edge_ids = np.arange(len(src_node_ids))
    # edge_ids = edge_ids + 1

    # get a list of snapshot batches from the timestamps
    snapshot_indices = get_snapshot_batches(node_interact_times)
    train_mask, val_mask, test_mask = generate_splits(full_data,
                                                      val_ratio=val_ratio,
                                                      test_ratio=test_ratio,
                                                      )

    # TODO: Actual edge features should be loaded!
    # edge_raw_features = np.zeros((src_node_ids.shape[0], 1))
    edge_raw_features = np.random.randn(src_node_ids.shape[0], 1)
    if len(edge_raw_features.shape) == 1:
        edge_raw_features = edge_raw_features[:, np.newaxis]

    num_nodes = len(set(src_node_ids) | set(dst_node_ids))
    max_idx = max(src_node_ids.max(), dst_node_ids.max()) + 1
    min_idx = min(src_node_ids.min(), dst_node_ids.min())
    if min_idx == 0:
        print(
            f"INFO: {dataset_name}: node index starts from 0, increasing all indices by 1.")
        src_node_ids = src_node_ids + 1
        dst_node_ids = dst_node_ids + 1

    # convert to Torch tensors
    src_node_ids, dst_node_ids, node_interact_times, labels, edge_ids = \
        convert_to_torch_extended(
            src_node_ids, dst_node_ids, node_interact_times, labels, edge_ids)
    temporal_data = TemporalData(
        sources=src_node_ids,
        destinations=dst_node_ids,
        timestamps=node_interact_times,
        labels=labels,
        edge_ids=edge_ids,
    )

    # Process to format used by NAT/DyGLib
    MAX_FEAT_DIM = 172
    if 'node_feat' not in full_data.keys():
        # node_raw_features = np.zeros((num_nodes + 1, 1))
        node_raw_features = np.random.randn(num_nodes + 1, 1)
    else:
        node_raw_features = full_data['node_feat'].astype(np.float64)
        # deal with node features whose shape has only one dimension
        if len(node_raw_features.shape) == 1:
            node_raw_features = node_raw_features[:, np.newaxis]

    # add feature of padded node and padded edge
    node_raw_features = np.vstack([np.zeros(node_raw_features.shape[1])[
                                  np.newaxis, :], node_raw_features])
    edge_raw_features = np.vstack([np.zeros(edge_raw_features.shape[1])[
                                  np.newaxis, :], edge_raw_features])

    assert MAX_FEAT_DIM >= node_raw_features.shape[
        1], f'Node feature dimension in dataset {dataset_name} is bigger than {MAX_FEAT_DIM}!'
    assert MAX_FEAT_DIM >= edge_raw_features.shape[
        1], f'Edge feature dimension in dataset {dataset_name} is bigger than {MAX_FEAT_DIM}!'

    start_times = {
        'train': node_interact_times[train_mask][0].item(),
        'val': node_interact_times[val_mask][0].item(),
        'test': node_interact_times[test_mask][0].item(),
    }

    end_times = {
        'train': node_interact_times[train_mask][-1].item(),
        'val': node_interact_times[val_mask][-1].item(),
        'test': node_interact_times[test_mask][-1].item(),
    }

    return temporal_data, start_times, end_times, node_raw_features, edge_raw_features, max_idx, snapshot_indices
