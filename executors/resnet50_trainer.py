import os
import pickle

import cv2
import matplotlib.pyplot as plt
import nmslib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm

from configs.neptune_cfg import neptune_cfg
from dataloader.batch_sampler import RetrievalBatchSampler
from dataset.dataset import RetrievalDataset
from losses.triplet_loss import TripletLoss
from metrics.recall import recall_at_k
from models.resnet50 import Resnet50
from models.utils import set_bn_eval
from utils.neptune_logger import NeptuneLogger
from utils.visualization import prepare_img, plot_topn
from torchvision.utils import make_grid


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.start_epoch, self.global_step = 0, 0
        self.max_epoch = cfg.max_epoch

        self.__get_data()
        self.__get_model()

        self.logger = NeptuneLogger(neptune_cfg)

    def __get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.device == 'gpu' else "cpu")
        self.model = Resnet50(embed_size=self.cfg.embed_size, with_norm=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr )
        self.criterion = TripletLoss(alpha=self.cfg.alpha)

        print(self.device)
        print('number of trainable params:\t', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def __get_data(self):
        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_dataset = RetrievalDataset(self.cfg.data_dir, 'train', transforms=train_preprocess)
        self.train_dataloader = torch.utils.data.dataloader.DataLoader(
            self.train_dataset,
            batch_sampler=RetrievalBatchSampler(self.train_dataset, m=self.cfg.m, l=self.cfg.l))

        test_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.valid_dataset = RetrievalDataset(self.cfg.data_dir, 'train', transforms=test_preprocess)
        self.valid_dataloader = torch.utils.data.dataloader.DataLoader(
            self.valid_dataset, shuffle=False, batch_size=self.cfg.eval_batch_size, drop_last=False)

        self.test_dataset = RetrievalDataset(self.cfg.data_dir, 'test', transforms=test_preprocess)
        self.test_dataloader = torch.utils.data.dataloader.DataLoader(
            self.test_dataset, shuffle=False, batch_size=self.cfg.eval_batch_size, drop_last=False)

    def _dump_model(self, epoch):
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }
        if not os.path.exists(self.cfg.checkpoints_dir):
            os.makedirs(self.cfg.checkpoints_dir)
        path_to_save = os.path.join(self.cfg.checkpoints_dir, f'epoch-{epoch}.pt')
        torch.save(state_dict, path_to_save)

    def _load_model(self, epoch):
        path = os.path.join(self.cfg.checkpoints_dir, f"epoch-{epoch}.pt")
        start_epoch = 0
        try:
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.global_step = state_dict['global_step']
            start_epoch = state_dict['epoch']
        except Exception as e:
            print(e)
        return start_epoch

    def make_training_step(self, data, labels):
        pred = self.model(data.to(self.device))
        loss = self.criterion(pred, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), pred.detach().cpu()

    @torch.no_grad()
    def get_embeddings(self, loader):
        all_embed, all_labels, images = [], [], []
        self.model.eval()
        with torch.no_grad:
            for i, (batch_data, batch_labels) in enumerate(tqdm(loader, desc='Get embeddings')):
                output = self.model(batch_data.to(self.device))
                all_embed.append(output.detach().cpu().view(-1, output.size(-1)))
                all_labels.extend(batch_labels)
                images.append(cv2.resize(prepare_img(batch_data.cpu()), (50, 50)))
        return {
            'embeddings': torch.vstack(all_embed).numpy(),
            'labels':  torch.vstack(all_labels).numpy(),
            'images': np.stack(images)
        }

    def overfit(self):
        batch_data, batch_labels = next(iter(self.train_dataloader))
        losses = []
        self.model.apply(set_bn_eval)
        nrof_iterations = 1000
        for _iter in range(0, nrof_iterations):
            loss, pred = self.make_training_step(batch_data, batch_labels)

            print(f"step {_iter}/{nrof_iterations}:\t",
                  ',\t'.join(['{!s}: {:.4f}'.format(name, metric) for name, metric in zip(['loss'], [loss])]))
            losses.append(loss)
        plt.plot(np.arange(len(losses)), losses)
        plt.show()

    def fit(self):
        if self.cfg.epoch_to_load is not None:
            self.start_epoch = self._load_model(self.cfg.epoch_to_load)

        if self.cfg.evaluate_before_training:
            self.evaluate('test', self.start_epoch)

        for epoch in range(self.start_epoch, self.max_epoch):
            self.model.train()
            self.model.apply(set_bn_eval)
            pbar = tqdm(self.train_dataloader)
            for batch_data, batch_labels in pbar:
                loss, pred = self.make_training_step(batch_data, batch_labels)
                self.global_step += 1
                self.logger.log_metrics(['loss/train'], [loss], self.global_step)
                pbar.set_description(desc='[]: loss - {:.4f}'.format(epoch, loss))

            self.evaluate('test', epoch + 1)
            self._dump_model(epoch + 1)

    def evaluate(self, data_type='test', epoch=None):
        loader = self.valid_dataloader if data_type == 'valid' else self.test_dataloader

        max_k = max(self.cfg.eval_top_k)

        path_to_embeds = os.path.join(self.cfg.embeddings_path, f"{epoch}")
        if not os.path.exists(os.path.join(path_to_embeds, f"{data_type}_data.pickle")):
            os.makedirs(path_to_embeds, exist_ok=True)
            data = self.get_embeddings(loader)
            with open(os.path.join(path_to_embeds, f"{data_type}_data.pickle"), 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(os.path.join(path_to_embeds, f"{data_type}_data.pickle"), 'rb') as f:
                data = pickle.load(f)

        # all_embed, all_labels, images = data['embeddings'], data['labels'], data['images']
        all_embed, all_labels = data['embeddings'], data['labels']

        index = nmslib.init(method='hnsw', space='l2')
        index.addDataPointBatch(all_embed)
        index.createIndex(print_progress=True)
        ids, _ = list(zip(*index.knnQueryBatch(all_embed, k=max_k + 1)))
        ids = np.asarray(ids)[:, 1:]
        nn_labels = all_labels[ids].reshape(-1, max_k)
        recall = recall_at_k(nn_labels, np.vstack(all_labels), topk=self.cfg.eval_top_k)

        # ind = np.random.randint(0, len(all_labels), 3)
        # query_images = images[ind]
        # query_labels = all_labels[ind]
        # retrieval_images = images[ids[ind]]
        # retrieval_labels = nn_labels[ind]
        # plot_topn(query_images, query_labels, retrieval_images, retrieval_labels, k=5)

        # print(recall)
        # self.logger.log_metrics([f"{data_type}/r@{k}" for k in recall.keys()], list(recall.values()), step=epoch)

        print(f"evaluation on {data_type}")
        print(f"\t[{epoch}]:", ',\t'.join(["r@{} - {:.2%}".format(k, r) for (k, r) in recall.items()]))

        # label_img = np.concatenate([images[retrieval_set], images[query_set]])
        # metadata = pd.DataFrame(data={
        #     'class': retrieval_labels.tolist() + query_labels.tolist(),
        #     'set_type': ['retrieval']*len(retrieval_labels) + ['query']*len(query_labels)})
        # embeddings = np.concatenate([retrieval_embed, query_embed])
        # image_retrieval_tensorboard_projection(writer, embeddings, metadata, label_img,
        #                                        exp_name=f"{data_type}_{self.cfg.experiment_name}_step_{step}")


if __name__ == '__main__':
    from configs.train_cfg import cfg as train_cfg
    trainer = Trainer(train_cfg)
    # trainer.overfit()
    trainer.evaluate(data_type='valid', epoch=1)