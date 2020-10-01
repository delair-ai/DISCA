import logging
import os
import time
from glob import glob

import cv2 as cv
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from PIL import Image
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from semantic_segmentation.loaders.loaders import GTDataset, RGBDataset
from semantic_segmentation.trainer import Trainer
from semantic_segmentation.utils.image import (from_coord_to_patch, grouper,
                                               sliding_window)
from semantic_segmentation.utils.metrics import IoU, accuracy, f1_score


class ClassicTrainer(Trainer):
    def __init__(self, cfg, train=True, dataset=None):
        super(ClassicTrainer, self).__init__(cfg)
        if train:
            self.train_dataset = RGBDataset(dataset, self.cfg)
            self.gt_dataset = GTDataset(dataset, self.cfg, self.train_dataset.train_ids)
            logging.info(
                f"Train ids (len {len(self.train_dataset.imgs)}): {[os.path.basename(i) for i in self.train_dataset.imgs]}"
            )
            self.dataset = dataset
        test_dataset = RGBDataset(dataset, self.cfg, False)
        logging.info(
            f"Test ids (len {len(test_dataset.imgs)}): {[os.path.basename(i) for i in test_dataset.imgs]}"
        )
        self.metrics = pd.DataFrame(
            data={i: [] for i in [os.path.basename(i) for i in test_dataset.imgs]}
        ).T

    def train(self, epochs, pretrain_file=None):
        logging.info(
            "%s INFO: Begin training",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )

        iter_ = 0

        start_epoch, accu, iou, f1, train_loss, test_loss, losses = self._load_init(
            pretrain_file
        )
        loss_weights = torch.ones(
            self.cfg.N_CLASSES, dtype=torch.float32, device=self.device
        )
        if self.cfg.WEIGHTED_LOSS or self.cfg.REVOLVER_WEIGHTED:
            weights = self.gt_dataset.compute_frequency()
            if self.cfg.REVOLVER_WEIGHTED:
                self.train_dataset.set_sparsifier_weights(weights)
            if self.cfg.WEIGHTED_LOSS:
                loss_weights = (
                    torch.from_numpy(weights).type(torch.FloatTensor).to(self.device)
                )

        train_loader = self.train_dataset.get_loader(
            self.cfg.BATCH_SIZE, self.cfg.WORKERS
        )
        for e in tqdm(range(start_epoch, epochs + 1), total=epochs + 1 - start_epoch):
            logging.info(
                "\n%s Epoch %s",
                time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
                e,
            )
            self.net.train()
            steps_pbar = tqdm(
                train_loader, total=self.cfg.EPOCH_SIZE // self.cfg.BATCH_SIZE
            )
            for data in steps_pbar:
                features, labels = data
                self.optimizer.zero_grad()
                features = features.float().to(self.device)
                labels = labels.float().to(self.device)
                output = self.net(features)
                loss = CrossEntropyLoss(loss_weights)(output, labels.long())
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                iter_ += 1
                steps_pbar.set_postfix({"loss": loss.item()})
            train_loss.append(np.mean(losses[-1 * self.cfg.EPOCH_SIZE :]))
            loss, iou_, acc_, f1_ = self.test()
            test_loss.append(loss)
            accu.append(acc_)
            iou.append(iou_ * 100)
            f1.append(f1_ * 100)
            del (loss, iou_, acc_)
            if e % 5 == 0:
                self._save_net(e, accu, iou, f1, train_loss, test_loss, losses)
            self.scheduler.step()
        pretrain_file = self._save_net(epochs, accu, iou, f1, train_loss, test_loss, losses, False)
        return pretrain_file


    def _infer_image(self, stride, *data, annots=None, full_pred=True):
        """infer one image"""
        with torch.no_grad():
            img = data[0]
            pred = np.zeros(img.shape[1:] + (self.cfg.N_CLASSES,))
            occur_pix = np.zeros((*img.shape[1:], 1))
        if full_pred:
            for coords in grouper(
                1,
                sliding_window(img, step=stride, window_size=self.cfg.WINDOW_SIZE),
            ):
                data_patches = from_coord_to_patch(img, coords, self.device)
                outs = self.net(data_patches).data.cpu().numpy()

                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x : x + w, y : y + h] += out
                    occur_pix[x : x + w, y : y + h, :] += 1
        else:
            for coords in grouper(
                    1,
                    sliding_window(img, step=stride, window_size=self.cfg.WINDOW_SIZE),
                ):
                data_patches = from_coord_to_patch(img, coords, self.device)
                annots_patches = from_coord_to_patch(annots, coords, self.device)
                if torch.sum(annots_patches) == 0:
                    continue
                outs = self.net(data_patches).data.cpu().numpy()

                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x : x + w, y : y + h] += out
                    occur_pix[x : x + w, y : y + h, :] += 1
        return pred, occur_pix[..., 0]

    def test(self, sparsity=0, id_class=None, initial_file=None):
        """Test the network on images.
        Args:
            sparsity (int): Number of clicks generated per image. Defalut: None
            id_class (int): class id of the newly sampled click.
                            Only used if sparsity >= 0. 
        """
        logging.info(
            "%s INFO: Begin testing",
            time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime()),
        )
        self.net.eval()
        loss, acc, iou, f1 = (
            [],
            [],
            [],
            [],
        )  # will contain the metric and loss calculated for each image
        test_dataset = RGBDataset(
            self.dataset,
            self.cfg,
            False,
            sparsity,
            id_class,
        )
        test_images = test_dataset.get_loader(1, self.cfg.TEST_WORKERS)
        stride = self.cfg.STRIDE
        modif_pxls = 0
        wrong_pxls = 0
        for idx, data in tqdm(
            zip(test_dataset.test_ids, test_images), total=len(test_dataset.test_ids)
        ):
            if initial_file is not None and self.cfg.CONTINUAL:
                net_filename = initial_file.replace('.', f'_{idx}_{self.cfg.ext}.')
                net_filename = os.path.join(os.path.dirname(net_filename), "tmp", os.path.basename(net_filename))
                if not os.path.exists(os.path.dirname(net_filename)):
                    os.mkdir(os.path.dirname(net_filename))
                if sparsity > 1:
                    checkpoint = torch.load(net_filename)
                    self.net.load_state_dict(checkpoint["net"])
            data = [i.squeeze(0) for i in data]
            img = data[0]
            gt = data[-1].cpu().numpy()
            data = data[:-1]
            raw_annots = img[self.cfg.IN_CHANNELS:self.cfg.IN_CHANNELS+self.cfg.N_CLASSES].clone()
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
            sparse_gt = [cv.dilate(i.cpu().numpy().astype(np.uint8), kernel)  for i in raw_annots]
            sparse_gt = [cv.distanceTransform(i.astype(np.uint8), cv.DIST_L2, 3) for i in sparse_gt]
            sparse_gt = np.stack(sparse_gt)
            img[self.cfg.IN_CHANNELS:self.cfg.IN_CHANNELS+self.cfg.N_CLASSES] = torch.from_numpy(sparse_gt).to(self.device)

            # Initial predictiton
            full_pred = sparsity == 0
            annots = torch.sum(raw_annots, 0).unsqueeze(0) != torch.unsqueeze(data[1], 0) if not full_pred else None
            pred_, occur_pix = self._infer_image(stride, *data, annots=annots, full_pred=full_pred)
            
            # Training if interactive learning
            if initial_file is not None and sparsity != 0 and self.cfg.CONTINUAL:
                lr = self.cfg.INTERACTIVE_LR
                continual_optimizer = optim.SGD(self.net.parameters(), lr)
                if sparsity > 1:
                    continual_optimizer.load_state_dict(checkpoint["optimizer"])
                state = self.update_weights(data, pred_, raw_annots.to(bool), annots, continual_optimizer)
                torch.save(state, net_filename)
                if sparsity % 10 == 0 and sparsity > 0:
                    torch.save(state, net_filename.replace('.', f"_{sparsity}sparsity."))

                # New prediction
                pred_, _ = self._infer_image(stride, *data)

            if sparsity == 0:
                loss.append(
                    CrossEntropyLoss()(
                        torch.from_numpy(np.expand_dims(pred_.transpose((2, 0, 1)), axis=0)),
                        torch.from_numpy(np.expand_dims(gt, axis=0)).long(),
                    ).item())
            pred = np.argmax(pred_, axis=-1)
            if not self.cfg.CONTINUAL and sparsity > 0:
                pred[occur_pix == 0] = data[2][occur_pix == 0]
            del occur_pix

            # Compute metrics
        metric_iou = IoU(
            pred, gt, self.cfg.N_CLASSES, all_iou=(sparsity is not None)
        )
        metric_f1 = f1_score(
            pred, gt, self.cfg.N_CLASSES, all=(sparsity is not None)
        )
        if sparsity is not None:
            metric_iou, all_iou = metric_iou
            metric_f1, all_f1, weighted_f1 = metric_f1
        metric_acc = accuracy(pred, gt)
        acc.append(metric_acc)
        iou.append(metric_iou)
        f1.append(metric_f1)
        if sparsity is not None:
            file_name = os.path.basename(
                sorted(glob(os.path.join(self.dataset, "gts", "*")))[idx]
            )
            name = os.path.join(
                self.cfg.SAVE_FOLDER,
                "tmp",
                "preds",
                self.cfg.ext + self.cfg.NET_NAME + file_name,
            )
            self.metrics.loc[file_name, f"{sparsity}_acc"] = metric_acc
            self.metrics.loc[file_name, f"{sparsity}_IoU"] = metric_iou
            self.metrics.loc[file_name, f"{sparsity}_F1"] = metric_f1
            self.metrics.loc[file_name, f"{sparsity}_F1_weighted"] = weighted_f1
            for c, i in enumerate(all_iou):
                self.metrics.loc[file_name, f"{sparsity}_IoU_class_{c}"] = i
            for c, i in enumerate(all_f1):
                self.metrics.loc[file_name, f"{sparsity}_F1_class_{c}"] = i
            if os.path.exists(name):
                old_pred = rio.open(name).read(1)
                diff = np.sum(old_pred != pred)
                modif_pxls += diff
                logging.info("%s: modified pixels: %s", file_name, diff)
                logging.info("IoU: %s", metric_iou)
                logging.info("F1: %s", metric_f1)
                wrongs = np.sum(np.bitwise_and(old_pred != pred, pred != gt))
                self.metrics.loc[file_name, f"{sparsity}_wrong_pxls"] = wrongs
                self.metrics.loc[file_name, f"{sparsity}_good_pxls"] = diff - wrongs
                wrong_pxls += wrongs

            dataset_name = os.path.basename(self.dataset)
            csv_name = "{}_{}{}.csv".format(
                os.path.join(self.cfg.SAVE_FOLDER, self.cfg.NET_NAME),
                dataset_name,
                self.cfg.ext,
            )
            self.metrics.to_csv(csv_name)
            Image.fromarray(pred.astype(np.uint8)).save(name)
        #  Update logger
        if sparsity is not None:
            logging.info(
                "Total modified pixels: %s", modif_pxls / len(test_dataset.test_ids)
            )
            logging.info(
                "Wrong modified pixels: %s", wrong_pxls / len(test_dataset.test_ids)
            )
        logging.info("Mean IoU : " + str(np.nanmean(iou)))
        logging.info("Mean accu : " + str(np.nanmean(acc)))
        logging.info("Mean F1 : " + str(np.nanmean(f1)))
        return np.mean(loss), np.nanmean(iou), np.mean(acc), np.mean(f1)

    def _load_init(self, pretrain_file):
        if pretrain_file:
            checkpoint = torch.load(pretrain_file)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.net.load_state_dict(checkpoint["state_dict"])
            train_loss = checkpoint["train_loss"]
            test_loss = checkpoint["test_loss"]
            start_epoch = checkpoint["epoch"]
            losses = checkpoint["losses"]
            accu = checkpoint["accu"]
            iou = checkpoint["iou"]
            f1 = checkpoint["f1"]
            logging.info(
                "Loaded checkpoint '{}' (epoch {})".format(
                    pretrain_file, checkpoint["epoch"]
                )
            )
        else:
            start_epoch = 1
            train_loss = []
            test_loss = []
            losses = []
            accu = []
            iou = []
            f1 = []
        return start_epoch, accu, iou, f1, train_loss, test_loss, losses

    def update_weights(self, img, initial_pred, annots, annots_diff, optimizer):
        steps = self.cfg.INTERACTIVE_STEPS
        sparse_target = torch.full([*annots.shape[-2:]], -1, dtype=torch.long, device=self.device)
        for i in range(self.cfg.N_CLASSES):
            sparse_target[annots[i]] = i
        sparse_target = sparse_target.unsqueeze(0)
        data = img[0]
        initial_pred = initial_pred.transpose((2, 0, 1))
        for iteration in range(steps):
            for coords in grouper(
                    1,
                    sliding_window(data, step=self.cfg.STRIDE, window_size=self.cfg.WINDOW_SIZE),
                ):
                optimizer.zero_grad()
                annots_patches = from_coord_to_patch(annots, coords, "cpu")
                if torch.sum(annots_patches) == 0:
                    continue
                data_patches = from_coord_to_patch(data, coords, self.device).float()
                data_patches[:, self.cfg.IN_CHANNELS:] = 0
                target = from_coord_to_patch(sparse_target, coords, self.device).long()
                target = target[0]
                pred = self.net(data_patches)
                
                ini_pred = from_coord_to_patch(initial_pred, coords, self.device)
                loss = CrossEntropyLoss(ignore_index=-1)(pred, target)
                reg = nn.L1Loss(reduction="none")(pred, ini_pred)
                reg = torch.mean(reg)
                loss = loss + reg
                loss.backward()
                optimizer.step()
        state = {
            "optimizer": optimizer.state_dict(),
            "net": self.net.state_dict(),
            }
        return state
