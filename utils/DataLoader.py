from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd

# TGB imports
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


tgb_data_num_nodes_map = {
    "tgbl-wiki": 9227,
    "tgbl-review": 352637,
    "tgbl-coin": 638486,
    "tgbl-comment": 994790,
    "tgbl-flight": 18143,
    "tgbn-trade": 255,
    "tgbn-genre": 992,
    "tgbn-reddit": 11068
}

tgb_data_num_edges_map = {
    "tgbl-wiki": 157474,
    "tgbl-review": 4873540,
    "tgbl-coin": 22809486,
    "tgbl-comment": 44314507,
    "tgbl-flight": 67169570,
    "tgbn-trade": 507497,
    "tgbn-genre": 17858395,
    "tgbn-reddit": 27174118
}

def get_link_prediction_tgb_data(dataset_name: str, 
                                 train_time_gran: str='ct', eval_time_gran: str='ct', time_scale: str=None):
    """
    generate tgb data for link prediction task
    :param dataset_name (str): dataset name
    :train_time_gran (str): training time granularity
    :eval_time_gran (str): evaluation time granularity
    
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object), eval_neg_edge_sampler, eval_metric_name
    """
    # Load data and train val test split
    dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
    
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    
    eval_metric_name = dataset.eval_metric
    
    eval_neg_edge_sampler = dataset.negative_sampler
    if time_scale is not None:
        if eval_time_gran == 'dt':
            print("DEBUG: Loading Validation & Test negative samples from:")
            print(f"\tDEBUG: Val.: {dataset.root}/{dataset_name}_val_ns_" + time_scale + ".pkl")
            print(f"\tDEBUG: Test.: {dataset.root}/{dataset_name}_test_ns_" + time_scale + ".pkl")
            dataset.negative_sampler.load_eval_set(fname=f"{dataset.root}/{dataset_name}_val_ns_" + time_scale + ".pkl", split_mode="val")
            dataset.negative_sampler.load_eval_set(fname=f"{dataset.root}/{dataset_name}_test_ns_" + time_scale + ".pkl", split_mode="test")
        else:
            dataset.load_val_ns()
            dataset.load_test_ns()
    else:
        dataset.load_val_ns()
        dataset.load_test_ns()
    
    if time_scale is not None:
        print(f"DEBUG: Load DTDG timestamps for all edges; Timestep: {time_scale}")
        relative_path = './dtdg_timestamps/' 
        time_scale = time_scale 
        ts_file = relative_path + f"{dataset_name}_ts_" + time_scale + ".csv"
        dtdg_ts = np.genfromtxt(ts_file, delimiter=',', dtype=int)
        print("DEBUG: DTDG Shape:", dtdg_ts.shape)
        
        # Whether to train or evaluate with DT or CT
        print(f"INFO: Over-writting timestamps...")
        if train_time_gran == 'dt':
            dataset.full_data['timestamps'][train_mask] = dtdg_ts[train_mask]
        if eval_time_gran == 'dt':
            dataset.full_data['timestamps'][val_mask] = dtdg_ts[val_mask]
            dataset.full_data['timestamps'][test_mask] = dtdg_ts[test_mask]
        

    # process the data to the required format
    data = dataset.full_data
    src_node_ids = data['sources'].astype(np.longlong)
    dst_node_ids = data['destinations'].astype(np.longlong)
    node_interact_times = data['timestamps'].astype(np.float64)
    edge_ids = data['edge_idxs'].astype(np.longlong)
    labels = data['edge_label']
    edge_raw_features = data['edge_feat'].astype(np.float64)
    
    # deal with edge features whose shape has only one dimension
    if len(edge_raw_features.shape) == 1:
        edge_raw_features = edge_raw_features[:, np.newaxis]
    # currently, we do not consider edge weights
    # edge_weights = data['w'].astype(np.float64)

    num_edges = edge_raw_features.shape[0]
    assert num_edges == tgb_data_num_edges_map[dataset_name], 'Number of edges are not matched!'

    # union to get node set
    num_nodes = len(set(src_node_ids) | set(dst_node_ids))
    assert num_nodes == tgb_data_num_nodes_map[dataset_name], 'Number of nodes are not matched!'

    assert src_node_ids.min() == 0 or dst_node_ids.min() == 0, "Node index should start from 0!"
    assert edge_ids.min() == 0 or edge_ids.min() == 1, "Edge index should start from 0 or 1!"
    # we notice that the edge id on the datasets (except for tgbl-wiki) starts from 1, so we manually minus the edge ids by 1
    if edge_ids.min() == 1:
        print(f"Manually minus the edge indices by 1 on {dataset_name}")
        edge_ids = edge_ids - 1
    assert edge_ids.min() == 0, "After correction, edge index should start from 0!"

    # note that in our data preprocess pipeline, we add an extra node and edge with index 0 as the padded node/edge for convenience of model computation,
    # therefore, for TGB, we also manually add the extra node and edge with index 0
    src_node_ids = src_node_ids + 1
    dst_node_ids = dst_node_ids + 1
    edge_ids = edge_ids + 1

    MAX_FEAT_DIM = 172
    if 'node_feat' not in data.keys():
        node_raw_features = np.zeros((num_nodes + 1, 1))
    else:
        node_raw_features = data['node_feat'].astype(np.float64)
        # deal with node features whose shape has only one dimension
        if len(node_raw_features.shape) == 1:
            node_raw_features = node_raw_features[:, np.newaxis]

    # add feature of padded node and padded edge
    node_raw_features = np.vstack([np.zeros(node_raw_features.shape[1])[np.newaxis, :], node_raw_features])
    edge_raw_features = np.vstack([np.zeros(edge_raw_features.shape[1])[np.newaxis, :], edge_raw_features])

    assert MAX_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {MAX_FEAT_DIM}!'
    assert MAX_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {MAX_FEAT_DIM}!'
    
    # split the data
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask], edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])
    
    # remove duplicated edges if DT evaluation is desired
    if time_scale is not None:  
        if eval_time_gran == 'dt':
            print("INFO: Removing duplicated edges for `val_data`...")
            val_data = remove_duplicate_edges(val_data)
            
            print("INFO: Removing duplicated edges for `test_data`...")
            test_data = remove_duplicate_edges(test_data)
            
    full_data = Data(src_node_ids=np.concatenate((train_data.src_node_ids, val_data.src_node_ids, test_data.src_node_ids), axis=0), 
                     dst_node_ids=np.concatenate((train_data.dst_node_ids, val_data.dst_node_ids, test_data.dst_node_ids), axis=0), 
                     node_interact_times=np.concatenate((train_data.node_interact_times, val_data.node_interact_times, test_data.node_interact_times), axis=0), 
                     edge_ids=np.concatenate((train_data.edge_ids, val_data.edge_ids, test_data.edge_ids), axis=0), 
                     labels=np.concatenate((train_data.labels, val_data.labels, test_data.labels), axis=0))

    print("INFO: The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("INFO: The training dataset has {} interactions, involving {} different nodes".format(train_data.num_interactions, train_data.num_unique_nodes))
    print("INFO: The validation dataset has {} interactions, involving {} different nodes".format(val_data.num_interactions, val_data.num_unique_nodes))
    print("INFO: The test dataset has {} interactions, involving {} different nodes".format(test_data.num_interactions, test_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name


def remove_duplicate_edges(data_split: dict):
    """
    remove the duplicated edges
    """
    src = data_split.src_node_ids
    dst = data_split.dst_node_ids
    ts = data_split.node_interact_times
    e_idx = data_split.edge_ids
    label = data_split.labels
    
    query = np.stack([src, dst, ts], axis=0)
    uniq, idx = np.unique(query, axis=1, return_index=True)
    print(f"\tINFO: original number of edges: {query.shape[1]}, number of duplicated edges: {uniq.shape[1]}")
    
    uniq_data_split = Data(src_node_ids=src[idx], dst_node_ids=dst[idx], 
                           node_interact_times=ts[idx], edge_ids=e_idx[idx], labels=label[idx])
    return uniq_data_split
    

def get_link_pred_data_TRANS_TGB(dataset_name: str):
    """
    generate data for link prediction task (NOTE: transductive dynamic link prediction)
    load the data with the help of TGB and generate required format for DyGLib
    :param dataset_name: str, dataset name
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172  # a specific setting for consistency among baselines
 
    # data loading
    dataset = PyGLinkPropPredDataset(name=dataset_name, root="datasets")
    data = dataset.get_TemporalData()
    # get split masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    # get split data
    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    # Load data and train val test split
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    edge_raw_features =  data.msg.numpy()
    node_raw_features = np.zeros((data.dst.size(0), NODE_FEAT_DIM))


    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], "Unaligned feature dimensions after feature padding!"

    src_node_ids = (data.src.numpy()+1).astype(np.longlong)
    dst_node_ids = (data.dst.numpy()+1).astype(np.longlong)
    node_interact_times = data.t.numpy().astype(np.float64)
    edge_ids = np.array([i for i in range(1, len(data.src)+1)]).astype(np.longlong)
    labels = data.y.numpy()

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, 
                    node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    print("INFO: The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("INFO: The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("INFO: The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("INFO: The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, dataset


def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return:
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], 172 - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], 172 - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], "Unaligned feature dimensions after feature padding!"

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.long)
    dst_node_ids = graph_df.i.values.astype(np.long)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.long)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data
