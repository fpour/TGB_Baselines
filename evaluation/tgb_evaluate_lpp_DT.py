import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math

from utils.utils import NeighborSampler


def query_pred_edge_batch(model_name: str, model: nn.Module,
                          src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray,
                          edges_are_positive: bool, num_neighbors: int, time_gap: int):
    """
    query the prediction probabilities for a batch of edges
    """
    if model_name in ['TGAT', 'CAWN', 'TCL']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                              dst_node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times,
                                                              num_neighbors=num_neighbors)

    elif model_name in ['JODIE', 'DyRep', 'TGN']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                              dst_node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times,
                                                              edge_ids=edge_ids,
                                                              edges_are_positive=edges_are_positive,
                                                              num_neighbors=num_neighbors)

    elif model_name in ['GraphMixer']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                              dst_node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times,
                                                              num_neighbors=num_neighbors,
                                                              time_gap=time_gap)

    elif model_name in ['DyGFormer']:
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, node_feat_dim)
        batch_src_node_embeddings, batch_dst_node_embeddings = \
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                              dst_node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times)

    else:
        raise ValueError(f"Wrong value for model_name {model_name}!")
        batch_src_node_embeddings, batch_dst_node_embeddings = None, None

    return batch_src_node_embeddings, batch_dst_node_embeddings


def eval_LPP_DT(model_name: str, model: nn.Module, device, neighbor_sampler: NeighborSampler,
                negative_sampler, temporal_data, snapshot_indices, start_times, end_times,
                evaluator, metric: str = "mrr", split_mode: str = "test",
                num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task for Discrete Time Dynamic Graphs
    """
    eval_snapshot_indices = range(
        start_times[split_mode], end_times[split_mode] + 1)

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate metrics
        evaluate_metrics = []
        evaluate_snapshot_idx_tqdm = tqdm(eval_snapshot_indices, ncols=120)
        for snap_idx in evaluate_snapshot_idx_tqdm:
            idx_start = snapshot_indices[snap_idx][0]
            idx_end = snapshot_indices[snap_idx][1]

            # temporal_data is Tensor, the model works best with np.ndarray
            src_node_ids = temporal_data.sources[idx_start:idx_end].clone(
            ).numpy().astype(np.longlong)
            dst_node_ids = temporal_data.destinations[idx_start:idx_end].clone(
            ).numpy().astype(np.longlong)
            node_interact_times = temporal_data.timestamps[idx_start:idx_end].clone(
            ).numpy().astype(np.float64)
            edge_ids = temporal_data.edge_ids[idx_start:idx_end].clone(
            ).numpy().astype(np.longlong)

            neg_batch_list = negative_sampler.query_batch(src_node_ids, dst_node_ids, node_interact_times,
                                                          split_mode=split_mode)
            num_negative_samples_per_node = [
                len(per_neg_batch) for per_neg_batch in neg_batch_list]

            pos_prob_list, neg_prob_list = [], []
            for idx, neg_batch in enumerate(neg_batch_list):
                batch_neg_src_node_ids = torch.full(
                    (len(neg_batch),), src_node_ids[idx], device=device).cpu().numpy()
                batch_neg_dst_node_ids = torch.from_numpy(np.array(neg_batch)).to(
                    dtype=torch.long, device=device).cpu().numpy()
                batch_neg_node_interact_times = torch.full(
                    (len(neg_batch),), node_interact_times[idx], device=device).cpu().numpy()

                # negative edges
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    query_pred_edge_batch(model_name=model_name, model=model,
                                          src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids,
                                          node_interact_times=batch_neg_node_interact_times, edge_ids=None,
                                          edges_are_positive=False, num_neighbors=num_neighbors, time_gap=time_gap)

                # one positive edge
                batch_pos_src_node_embeddings, batch_pos_dst_node_embeddings = \
                    query_pred_edge_batch(model_name=model_name, model=model,
                                          src_node_ids=np.array([src_node_ids[idx]]), dst_node_ids=np.array([dst_node_ids[idx]]),
                                          node_interact_times=np.array([node_interact_times[idx]]), edge_ids=np.array([edge_ids[idx]]),
                                          edges_are_positive=True, num_neighbors=num_neighbors, time_gap=time_gap)

                # get positive and negative probabilities
                positive_probabilities = model[1](input_1=batch_pos_src_node_embeddings,
                                                  input_2=batch_pos_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings,
                                                  input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                pos_prob_list.append(positive_probabilities.cpu().numpy())
                neg_prob_list.append(negative_probabilities.cpu().numpy())

            positive_probabilities = np.array([
                item for sublist in pos_prob_list for item in sublist])
            negative_probabilities = np.array([
                item for sublist in neg_prob_list for item in sublist])
            for sample_idx in range(len(src_node_ids)):
                neg_start_idx = sum(
                    num_negative_samples_per_node[:sample_idx]) - 1
                neg_end_idx = neg_start_idx + \
                    num_negative_samples_per_node[sample_idx]  # inclusive
                # compute metric
                input_dict = {
                    # use slices instead of index to keep the dimension of y_pred_pos
                    "y_pred_pos": positive_probabilities[sample_idx: sample_idx + 1],
                    "y_pred_neg": negative_probabilities[neg_start_idx: neg_end_idx],
                    "eval_metric": [metric],
                }
                evaluate_metrics.append(evaluator.eval(input_dict)[metric])

    avg_perf_metric = float(np.mean(np.array(evaluate_metrics)))

    return avg_perf_metric
