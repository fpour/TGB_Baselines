import os
import numpy as np
import math
import tgx
import tgb
import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.undirected import to_undirected


def get_edges(edge_index_list: list) -> list:
    """
    obtain undirected edges and transpose the edge index
    #! appending snapshots to a long list is not speed efficient
    parameter:
        edge_index_list: list of edge_index arrays, where each array is the edges in a given snapshot, can change to dictionary of edge index arrays
    output:
        undirected_edge_list: list of undirected edge_index arrays, where each array is the undirected edges in a given snapshot

    output usage syntax:
    data['edge_index_list'][snapshot_idx], this can just be a dictionary
    """
    undirected_edge_list = {}
    # idx = 0  #if there is time gap, the empty snapshots will be skipped
    # ! adapt to assume variable start index
    idx = min(list(edge_index_list.keys()))
    for i in range(idx, idx + len(edge_index_list)):
        if (i not in edge_index_list):
            raise Exception(
                "There are time gap in the dataset, encountered empty snapshot, please use coarser time granularity.")
            # undirected_edge_list[idx] = None
            # continue #empty snapshots
        else:
            edge_index, _ = remove_self_loops(
                torch.from_numpy(np.array(edge_index_list[i])))  # remove self-loop
            # convert to undirected/bi-directed edge_index
            undirected_edge_list[idx] = to_undirected(edge_index)
            idx += 1
    return undirected_edge_list


def load_dtdg(dataset_name: str,
              time_scale: str,
              verbose: bool = True):
    r"""
    load a DTDG dataset from built-in TGX datasets
    Parameters:
        dataset_name: name of the dataset
        time_scale: time scale for discretization
        verbose: print out the dataset information
    Output:
        dtdg: discretized temporal dynamic graph
        ts_list: list of timestamps
    """
    dataset_name = dataset_name.lower()
    if (dataset_name == "uci"):
        dataset = tgx.builtin.uci()
    elif (dataset_name == "canparl"):
        dataset = tgx.builtin.canparl()
    elif (dataset_name == "unvote"):
        dataset = tgx.builtin.unvote()
    elif (dataset_name == "uslegis"):
        dataset = tgx.builtin.uslegis()
    elif (dataset_name == "untrade"):
        dataset = tgx.builtin.untrade()
    elif (dataset_name == "enron"):
        dataset = tgx.builtin.enron()
    elif (dataset_name == "contacts"):
        dataset = tgx.builtin.contacts()
    elif (dataset_name == "social_evo"):
        dataset = tgx.builtin.social_evo()
    elif (dataset_name == "mooc"):
        dataset = tgx.builtin.mooc()
    else:
        raise ValueError("ERROR: unsupported dataset in TGX:  ", dataset_name)

    # data loading of dtdg dataset
    ctdg = tgx.Graph(dataset)
    time_scale = time_scale
    dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
    if (verbose):
        print("processing dataset: ", dataset_name)
        print("discretize to ", time_scale)
        print("there is time gap, ", dtdg.check_time_gap())
    return dtdg, ts_list


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    print("INFO: log_folder:", log_folder)
    return log_folder


def process_edges(edge_index_list: dict,
                  num_nodes: int,
                  ts_map: list = None,
                  keep_original: bool = False):
    r"""
    return a Data object for a split of a TGB dataset
    Parameters:
        edge_index_list: dict, keys are integer timestamps, values are edge_index
        num_nodes: int, number of nodes in the graph
        ts_map: list, element are the corresponding unix timestamp of the snapshots
    """
    pos_undirected_edges = get_edges(edge_index_list)

    if (keep_original):
        data = {
            'edge_index': pos_undirected_edges,
            # total number of nodes across all split; this is the same value for each split
            'num_nodes': num_nodes,
            'time_length': len(pos_undirected_edges),
            'ts_map': ts_map,
            'original_edges': edge_index_list,
        }
    else:
        data = {
            'edge_index': pos_undirected_edges,
            # total number of nodes across all split; this is the same value for each split
            'num_nodes': num_nodes,
            'time_length': len(pos_undirected_edges),
            'ts_map': ts_map,
        }
    return data


# ! use to load snapshots from TGB dataset
def TGB_data_discrete_processing(dataset_name: str,
                                 time_scale: str,
                                 split_mode: str = "train"):
    r"""
    process a TGB dataset with discretization
    parameters:
        dataset_name: name of the dataset
        time_scale: time scale for discretization
    Output:
    """

    # * for discrete models, we only use the discretized edges for training, the test phase will broadcast the predictions
    data_file = dataset_name + "_" + time_scale + "_tgx.pkl"

    if os.path.isfile(data_file):
        print("--------------------")
        print("Loading TGX graph", data_file)
        print("--------------------")
        dtdg = tgb.utils.utils.load_pkl(data_file)
    else:
        # * generate your own discrete timestamps
        # only keep the training snapshots
        tgx_dataset = tgx.tgb_data(dataset_name)
        if (split_mode == "train"):
            mask = tgx_dataset.train_mask
        elif (split_mode == "val"):
            mask = tgx_dataset.val_mask
        elif (split_mode == "test"):
            mask = tgx_dataset.test_mask
        # here only looking at the edges
        tgx_dataset.data = tgx_dataset.data[mask]
        ctdg = tgx.Graph(tgx_dataset)

        dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
        # dtdg.shift_time_to_zero()

    """
    the number of snapshots in ts_list is different than snapshots
    because snapshots skips over empty snapshots
    """
    snapshots = {}
    # dtdg.data format is {ts: {(u,v):1}}
    for ts in dtdg.data.keys():
        if isinstance(dtdg.data[ts], dict):
            edges = list(dtdg.data[ts].keys())
        else:
            edges = dtdg.data[ts]
        edges = np.array(edges).astype(int)
        # ! edges are in shape (num_edges,2) need to convert to (2, num_edges)
        edges = np.swapaxes(edges, 0, 1)
        snapshots[ts] = edges

    # num_nodes = dtdg.total_nodes() + 1 #this calculates the # of unique nodes
    # this calculates max node ID in the dataset
    num_nodes = int(dtdg.max_nid()) + 1

    ts_list = list(set(ts_list))
    ts_list.sort()
    return snapshots, num_nodes, ts_list


# ! use to load snapshots from TGB dataset
def TGB_process_all(dataset_name: str,
                    time_scale: str, ):
    r"""
    process the entire TGB dataset with discretization
    parameters:
        dataset_name: name of the dataset
        time_scale: time scale for discretization
    """

    # * generate your own discrete timestamps
    # only keep the training snapshots
    tgx_dataset = tgx.tgb_data(dataset_name)
    # tgx_dataset.data = tgx_dataset.data #here only looking at the edges
    ctdg = tgx.Graph(tgx_dataset)
    dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
    """
    the number of snapshots in ts_list is different than snapshots
    because snapshots skips over empty snapshots
    """
    snapshots = {}
    # dtdg.data format is {ts: {(u,v):1}}
    for ts in dtdg.data.keys():
        if isinstance(dtdg.data[ts], dict):
            edges = list(dtdg.data[ts].keys())
        else:
            edges = dtdg.data[ts]
        edges = np.array(edges).astype(int)
        # ! edges are in shape (num_edges,2) need to convert to (2, num_edges)
        edges = np.swapaxes(edges, 0, 1)
        snapshots[ts] = edges

    # this calculates max node ID in the dataset
    num_nodes = int(dtdg.max_nid()) + 1

    ts_list = list(set(ts_list))
    ts_list.sort()
    return snapshots, num_nodes, ts_list


def load_TGX_dataset(dataset_name: str,
                     time_scale: str, ):
    r"""
    load a built in TGX dataset from the DGB paper
    parameters:
        dataset_name: name of the dataset
        time_scale: time scale for discretization
    Output:
        train_data: training snapshots
        val_data: validation snapshots
        test_data: testing snapshots
    """
    dataset_name = dataset_name.lower()
    if (dataset_name == "uci"):
        dataset = tgx.builtin.uci()
    elif (dataset_name == "canparl"):
        dataset = tgx.builtin.canparl()
    elif (dataset_name == "unvote"):
        dataset = tgx.builtin.unvote()
    elif (dataset_name == "uslegis"):
        dataset = tgx.builtin.uslegis()
    elif (dataset_name == "untrade"):
        dataset = tgx.builtin.untrade()
    elif (dataset_name == "enron"):
        dataset = tgx.builtin.enron()
    elif (dataset_name == "contacts"):
        dataset = tgx.builtin.contacts()
    elif (dataset_name == "social_evo"):
        dataset = tgx.builtin.social_evo()
    elif (dataset_name == "mooc"):
        dataset = tgx.builtin.mooc()
    else:
        raise ValueError("ERROR: unsupported dataset in TGX:  ", dataset_name)

    ctdg = tgx.Graph(dataset)
    dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
    ts_list = list(dtdg.data.keys())
    ts_list.sort()

    val_ratio = 0.15
    test_ratio = 0.15
    # * generate val and test split
    val_time, test_time = list(
        np.quantile(
            ts_list,
            [(1 - val_ratio - test_ratio), (1 - test_ratio)],
        )
    )

    val_time = math.ceil(val_time)
    test_time = math.ceil(test_time)

    train_snapshots = {}
    val_snapshots = {}
    test_snapshots = {}
    # dtdg.data format is {ts: {(u,v):1}}

    for ts in ts_list:
        if isinstance(dtdg.data[ts], dict):
            edges = list(dtdg.data[ts].keys())
        else:
            edges = dtdg.data[ts]
        edges = np.array(edges).astype(int)
        # ! edges are in shape (num_edges,2) need to convert to (2, num_edges)
        edges = np.swapaxes(edges, 0, 1)
        assert edges.shape[0] == 2
        if (ts <= val_time):
            train_snapshots[ts] = edges
        elif (ts > val_time and ts <= test_time):
            val_snapshots[ts] = edges
        else:
            test_snapshots[ts] = edges

    # this calculates max node ID in the dataset
    num_nodes = int(dtdg.max_nid()) + 1
    print("there are ", dtdg.total_nodes(), " nodes in the dataset")
    print(" maximum node id is ", dtdg.max_nid())
    train_data = process_edges(train_snapshots, num_nodes, list(
        train_snapshots.keys()), keep_original=True)
    val_data = process_edges(val_snapshots, num_nodes, list(
        val_snapshots.keys()), keep_original=True)
    test_data = process_edges(test_snapshots, num_nodes, list(
        test_snapshots.keys()), keep_original=True)
    return train_data, val_data, test_data


"""
loading a TGB dataset based on a given discretization
"""


def load_TGB_dataset(dataset_name: str,
                     time_scale: str):
    r"""
    load a TGB dataset with discretization
    parameters:
        dataset_name: name of the dataset
        time_scale: time scale for discretization
    Output:
        train_data: training snapshots
        val_data: validation snapshots
        test_data: testing snapshots
    """

    # * maybe we should discrete together then split into the train, val snapshots
    all_snapshots_original, num_nodes, ts_list = TGB_process_all(
        dataset_name, time_scale)

    # ! need to have the snapshot ids start with 1, it will not start with 0 if the unix timestamp is absolute (meaning documenting real world time), will start at 0 if it is relative (start at 0 for its unix timestamp as well)
    if (min(list(all_snapshots_original.keys())) != 0):
        print("remap snapshot index to start from 0")
        all_snapshots = {}
        min_key = min(list(all_snapshots_original.keys()))
        keys = list(all_snapshots_original.keys())
        keys.sort()
        for key in sorted(keys):
            all_snapshots[key - min_key] = all_snapshots_original[key]
    else:
        all_snapshots = all_snapshots_original

    # * now we break down the snapshots into train, val, test
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    dataset = PyGLinkPropPredDataset(name=dataset_name, root="datasets")
    full_data = dataset.get_TemporalData()
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    train_data = full_data[train_mask]
    val_data = full_data[val_mask]
    test_data = full_data[test_mask]
    last_train_ts = train_data.t[-1]
    last_val_ts = val_data.t[-1]
    last_test_ts = test_data.t[-1]

    # ? find the last snapshots in train, val, the rest is in test
    train_snapshots = {}
    val_snapshots = {}
    test_snapshots = {}
    train_ts = {}
    val_ts = {}
    test_ts = {}

    for i in range(len(ts_list)):
        if ts_list[i] <= last_train_ts:
            train_snapshots[i] = all_snapshots[i]
            train_ts[i] = ts_list[i]
        elif ts_list[i] <= last_val_ts:
            val_snapshots[i] = all_snapshots[i]
            val_ts[i] = ts_list[i]
        else:
            test_snapshots[i] = all_snapshots[i]
            test_ts[i] = ts_list[i]

    assert len(train_snapshots) + len(val_snapshots) + len(test_snapshots) == len(
        all_snapshots), "all snapshots are accounted for"
    assert list(train_ts.keys()) == sorted(
        list(train_ts.keys())), "train timestamps are sorted"
    assert list(val_ts.keys()) == sorted(
        list(val_ts.keys())), "val timestamps are sorted"
    assert list(test_ts.keys()) == sorted(
        list(test_ts.keys())), "test timestamps are sorted"

    train_data = process_edges(train_snapshots, num_nodes, train_ts)
    val_data = process_edges(val_snapshots, num_nodes, val_ts)
    test_data = process_edges(test_snapshots, num_nodes, test_ts)

    return train_data, val_data, test_data


def loader(dataset='uci', time_scale=None):
    """
    loader function to check with dataset to be loaded
    """
    # if not cached, to process and cached
    print('INFO: data does not exits, processing ...')

    if dataset in ['uci', 'canparl', 'unvote', 'uslegis', 'untrade', 'contacts', 'enron', 'socialevo', 'lastfm',
                   'contacts', 'mooc', 'social_evo']:
        print("INFO: Loading TGX built-in DGB dataset: {}".format(dataset))
        train_data, val_data, test_data = load_TGX_dataset(dataset, time_scale)
        data = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
        }

    elif 'tgb' in dataset:
        train_data, val_data, test_data = load_TGB_dataset(
            dataset, time_scale=time_scale)
        print("INFO: Loading TGB dataset: {}".format(dataset))
        data = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
        }

    else:
        raise ValueError("ERROR: Undefined dataset!")
    return data
