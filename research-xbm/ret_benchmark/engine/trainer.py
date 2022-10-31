# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import datetime
import time

import numpy as np
import torch

from ret_benchmark.data.evaluations.eval import AccuracyCalculator
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.utils.metric_logger import MetricLogger
from ret_benchmark.utils.log_info import log_info
from ret_benchmark.modeling.xbm import XBM
from neptune_logger import NeptuneLogger
from neptune_cfg import neptune_cfg


def flush_log(writer, iteration):
    for k, v in log_info.items():
        if isinstance(v, np.ndarray):
            writer.add_histogram(k, v, iteration)
        else:
            writer.add_scalar(k, v, iteration)
    for k in list(log_info.keys()):
        del log_info[k]


def do_train(
    cfg,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    checkpointer,
    writer,
    device,
    checkpoint_period,
    arguments,
    logger, run_id
):
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = cfg.SOLVER.MAX_ITERS

    best_iteration = -1
    best_mapr = 0

    start_training_time = time.time()
    end = time.time()

    if cfg.XBM.ENABLE:
        logger.info(">>> use XBM")
        xbm = XBM(cfg)
    neptune_logger = NeptuneLogger(neptune_cfg, run_id)

    iteration = 0

    _train_loader = iter(train_loader)
    while iteration <= max_iter:
        try:
            images, targets, indices = _train_loader.next()
        except StopIteration:
            _train_loader = iter(train_loader)
            images, targets, indices = _train_loader.next()

        if (
            iteration % cfg.VALIDATION.VERBOSE == 0 or iteration == max_iter
        ):
            model.eval()
            logger.info("Validation")

            labels = val_loader[0].dataset.label_list
            labels = np.array([int(k) for k in labels])
            feats = feat_extractor(model, val_loader[0], logger=logger)
            ret_metric = AccuracyCalculator(include=(
                "precision_at_1",
                "mean_average_precision_at_r", "r_precision",
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "recall_at_10"
            ), exclude=())
            ret_metric = ret_metric.get_accuracy(feats, feats, labels, labels, True)
            mapr_curr = ret_metric['recall_at_1']
            for k, v in ret_metric.items():
                log_info[f"e_{k}"] = v

            scheduler.step(log_info[f"e_precision_at_1"])
            log_info["lr"] = optimizer.param_groups[0]["lr"]
            neptune_logger.log_metrics(['train/lr'], [log_info["lr"]], step=iteration)
            if mapr_curr > best_mapr:
                best_mapr = mapr_curr
                best_iteration = iteration
                logger.info(f"Best iteration {iteration}: {ret_metric}")
            else:
                logger.info(f"Performance at iteration {iteration:06d}: {ret_metric}")
            flush_log(writer, iteration)

            _recall = [ret_metric['recall_at_1'], ret_metric['recall_at_3'], ret_metric['recall_at_5'], ret_metric['recall_at_10']]
            k = [1, 3, 5, 10]
            names = [f"test/r@{kk}" for kk in k]
            neptune_logger.log_metrics(names, _recall, step=iteration)


            labels = val_loader[1].dataset.label_list
            labels = np.array([int(k) for k in labels])
            feats = feat_extractor(model, val_loader[1], logger=logger)
            ret_metric = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r", "r_precision", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10"), exclude=())
            ret_metric = ret_metric.get_accuracy(feats, feats, labels, labels, True)
            _recall = [ret_metric['recall_at_1'], ret_metric['recall_at_3'], ret_metric['recall_at_5'],
                       ret_metric['recall_at_10']]
            k = [1, 3, 5, 10]
            names = [f"valid/r@{kk}" for kk in k]
            neptune_logger.log_metrics(names, _recall, step=iteration)


        model.train()

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = targets.to(device)
        feats = model(images)

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION:
            xbm.enqueue_dequeue(feats.detach(), targets.detach())

        loss = criterion(feats, targets, feats, targets)
        log_info["batch_loss"] = loss.item()

        neptune_logger.log_metrics(['batch_loss/train'], [loss.item()], step=iteration)

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION:
            xbm_feats, xbm_targets = xbm.get()
            xbm_loss = criterion(feats, targets, xbm_feats, xbm_targets)
            log_info["xbm_loss"] = xbm_loss.item()
            loss = loss + cfg.XBM.WEIGHT * xbm_loss

            neptune_logger.log_metrics(['xbm_loss/train'], [xbm_loss.item()], step=iteration)

        neptune_logger.log_metrics(['loss/train'], [loss.item()], step=iteration)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time, loss=loss.item())
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.1f} GB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )

            log_info["loss"] = loss.item()
            flush_log(writer, iteration)

        if iteration % checkpoint_period == 0 and cfg.SAVE:
            checkpointer.save("model_{:06d}".format(iteration))
            pass

        del feats
        del loss

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    logger.info(f"Best iteration: {best_iteration :06d} | best MAP@R {best_mapr} ")
    writer.close()
