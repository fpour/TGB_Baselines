"""
Train a TG model and evaluate it with TGB package
NOTE:  The task is Transductive Dynamic Link Prediction
"""

import logging
import timeit
import time
import datetime
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler_pyg_TD, NegativeEdgeSampler_local
from utils.metrics import get_link_prediction_metrics
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

from tgb.linkproppred.evaluate import Evaluator
from evaluation.tgb_evaluate_lpp_DT import eval_LPP_DT
from utils.DataLoader_DT import get_lpp_data_DT
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler


def main():

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    temporal_data, start_times, end_times, node_raw_features, edge_raw_features, max_idx, snapshot_indices = \
        get_lpp_data_DT(args.dataset_name, args.time_scale,
                        args.val_ratio, args.test_ratio)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler_pyg_TD(temporal_data, start_times['train'], end_times['train'], snapshot_indices,
                                                         sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                         time_scaling_factor=args.time_scaling_factor, seed=0)  # train_data

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler_pyg_TD(temporal_data, start_times['train'], end_times['test'], snapshot_indices,
                                                        sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                        time_scaling_factor=args.time_scaling_factor, seed=1)  # full_data

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    train_start_index = snapshot_indices[start_times['train']][0]
    train_end_index = snapshot_indices[end_times['train']][1] + 1
    train_src_node_ids = temporal_data.sources[train_start_index:train_end_index].clone(
    ).numpy()
    train_dst_node_ids = temporal_data.destinations[train_start_index:train_end_index].clone(
    ).numpy()
    train_node_interact_times = temporal_data.timestamps[train_start_index:train_end_index].clone(
    ).numpy()

    train_neg_edge_sampler = NegativeEdgeSampler_local(
        src_node_ids=train_src_node_ids, dst_node_ids=train_dst_node_ids)

    # Set negative sampler for TGB-style evaluation
    eval_neg_edge_sampler = NegativeEdgeSampler(
        dataset_name=args.dataset_name, strategy="hist_rnd")

    # load negative samples for evaluation
    split_mode = 'val'
    eval_neg_edge_sampler.load_eval_set(
        fname=f"data/{args.dataset_name}/{args.dataset_name}_{split_mode}_ns.pkl", split_mode=split_mode)
    split_mode = 'test'
    eval_neg_edge_sampler.load_eval_set(
        fname=f"data/{args.dataset_name}/{args.dataset_name}_{split_mode}_ns.pkl", split_mode=split_mode)

    # evaluating with a TGB's evaluator
    metric = "mrr"  # NOTE: this is better to be set globally
    evaluator = Evaluator(name=args.dataset_name)

    for run in range(args.num_runs):
        start_run = timeit.default_timer()
        set_random_seed(seed=args.seed+run)
        # train_neg_edge_sampler.reset_random_state(seed=args.seed+run)

        args.save_model_name = f'{args.model_name}_{args.dataset_name}_timeScale_{args.time_scale}_seed_{args.seed}_run_{run}_DT'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        log_start_time = datetime.datetime.fromtimestamp(
            time.time()).strftime("%Y-%m-%d_%H:%M:%S")
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(log_start_time)}_DT.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f'Configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                    num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(
                    train_src_node_ids, train_dst_node_ids, train_node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                           neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                           model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift,
                                           src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                                           dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                          neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors,
                                          num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                         neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
                                         patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # define optimizer
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        # define the early stopping module
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()  # sigmoid should be applied explicitly
        # loss_func = nn.BCEWithLogitsLoss()  # since the link_predictor does not have a `sigmoid`

        # ================================================
        # ============== train & validation ==============
        # ================================================
        train_snapshot_indices = range(
            start_times['train'], end_times['train'] + 1)

        val_perf_list = []
        train_time_list, val_time_list, epoch_time_list = [], [], []
        for epoch in range(args.num_epochs):
            start_epoch = timeit.default_timer()
            start_train = timeit.default_timer()
            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_data_snap_tqdm = tqdm(train_snapshot_indices, ncols=120)
            for snap_idx in train_data_snap_tqdm:
                idx_start = snapshot_indices[snap_idx][0]
                idx_end = snapshot_indices[snap_idx][1]

                src_node_ids = temporal_data.sources[idx_start:idx_end].clone(
                ).numpy()
                dst_node_ids = temporal_data.destinations[idx_start:idx_end].clone(
                ).numpy()
                node_interact_times = temporal_data.timestamps[idx_start:idx_end].clone(
                ).numpy()
                edge_ids = temporal_data.edge_ids[idx_start:idx_end].clone(
                ).numpy

                _, neg_dst_node_ids = train_neg_edge_sampler.sample(
                    size=len(src_node_ids))
                neg_src_node_ids = src_node_ids

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                          dst_node_ids=dst_node_ids,
                                                                          node_interact_times=node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=neg_src_node_ids,
                                                                          dst_node_ids=neg_dst_node_ids,
                                                                          node_interact_times=node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=neg_src_node_ids,
                                                                          dst_node_ids=neg_dst_node_ids,
                                                                          node_interact_times=node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                          dst_node_ids=dst_node_ids,
                                                                          node_interact_times=node_interact_times,
                                                                          edge_ids=edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                          dst_node_ids=dst_node_ids,
                                                                          node_interact_times=node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=neg_src_node_ids,
                                                                          dst_node_ids=neg_dst_node_ids,
                                                                          node_interact_times=node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                elif args.model_name in ['DyGFormer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                          dst_node_ids=dst_node_ids,
                                                                          node_interact_times=node_interact_times)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=neg_src_node_ids,
                                                                          dst_node_ids=neg_dst_node_ids,
                                                                          node_interact_times=node_interact_times)
                else:
                    raise ValueError(
                        f"Wrong value for model_name {args.model_name}!")

                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = model[1](input_1=batch_src_node_embeddings,
                                                  input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings,
                                                  input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                predicts = torch.cat(
                    [positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(
                    positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(
                    predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_data_snap_tqdm.set_description(
                    f'Epoch: {epoch + 1}, train for the {snap_idx + 1}-th batch, train loss: {loss.item()}')

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            end_train = timeit.default_timer()
            train_time_list.append(end_train - start_train)

            # ==============================================
            # === validation
            # after one complete epoch, evaluate the model on the validation set
            start_val = timeit.default_timer()
            val_metric = eval_LPP_DT(model_name=args.model_name, model=model, device=args.device, neighbor_sampler=full_neighbor_sampler,
                                     negative_sampler=eval_neg_edge_sampler,
                                     temporal_data=temporal_data, snapshot_indices=snapshot_indices,
                                     start_times=start_times, end_times=end_times,
                                     evaluator=evaluator, metric=metric, split_mode='val',
                                     num_neighbors=args.num_neighbors, time_gap=args.time_gap)
            val_perf_list.append(val_metric)
            end_val = timeit.default_timer()
            val_time_list.append(end_val - start_val)

            epoch_time = timeit.default_timer() - start_epoch
            epoch_time_list.append(epoch_time)
            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}, elapsed time (s): {epoch_time:.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'Validation: {metric}: {val_metric: .4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = [(metric, val_metric, True)]
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        total_train_val_time = timeit.default_timer() - start_run
        logger.info(
            f'Total train & validation elapsed time (s): {total_train_val_time:.6f}')

        # ========================================
        # ============== Final Test ==============
        # ========================================
        start_test = timeit.default_timer()
        test_metric = val_metric = eval_LPP_DT(model_name=args.model_name, model=model, device=args.device, neighbor_sampler=full_neighbor_sampler,
                                               negative_sampler=eval_neg_edge_sampler,
                                               temporal_data=temporal_data, snapshot_indices=snapshot_indices,
                                               start_times=start_times, end_times=end_times,
                                               evaluator=evaluator, metric=metric, split_mode='test',
                                               num_neighbors=args.num_neighbors, time_gap=args.time_gap)
        test_time = timeit.default_timer() - start_test
        logger.info(f'Test elapsed time (s): {test_time:.4f}')
        logger.info(f'Test: {metric}: {test_metric: .4f}')

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "data": args.dataset_name,
            "model": args.model_name,
            "run": run,
            "seed": args.seed,
            "time_scale": args.time_scale,
            'train_time_gran': args.train_time_gran,
            'eval_time_gran': args.eval_time_gran,
            "train_time_list": train_time_list,
            "val_time_list": val_time_list,
            "epoch_time_list": epoch_time_list,
            "avg_train_time": np.mean(train_time_list),
            "avg_val_time": np.mean(val_time_list),
            "avg_epoch_time": np.mean(epoch_time_list),
            "total_train_val_time": total_train_val_time,
            f"validation {metric}": val_perf_list,
            "num_epoch": len(val_perf_list),
            f"test {metric}": test_metric,
            f"best validation {metric}": np.max(val_perf_list),
            "test_time": test_time,
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}_{args.time_scale}_DT"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.save_model_name}_{args.time_scale}_DT.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

        logger.info(
            f"run {run} total elapsed time (s): {timeit.default_timer() - start_run:.4f}")


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    main()
