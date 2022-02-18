import os
import torch
import time
from sklearn.model_selection import KFold
from collections import defaultdict
from tqdm import tqdm


import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.utils.functions import reset_weights


class BaseTrainer:
    """
    Base Trainer class used for training and validation. Should be wrapped
    by another class.
    """
    def __init__(self, model,
                 optimizer,
                 config,
                 metrics=None,
                 scheduler=None):
        """

        :param model: the model to be trained
        :param optimizer: optimizer used
        :param config: config file
        :param metrics: metrics to be reported
        :param scheduler: scheduler to be used (if applied)
        """
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        # the directory to save the model
        self.checkpoint_dir = config.checkpoint_dir
        # 'clip' is used for gradient clipping (if applied)
        self.clip = config.clip
        self.device = config.device
        # whether to skip validation process
        self.skip_val = config.skip_val
        # patience for early stopping
        self.patience = config.patience
        # wheter to apply early stopping or not
        self.early_stopping = config.early_stopping
        # dir to save logs for tensorboard
        self.logdir = config.logdir
        self.writer = SummaryWriter(self.logdir)
        # specified metrics to be reported
        self.metrics = metrics
        #  used for logging
        self._steps = 0
        self._train_logs = 0
        self._val_logs = 0
        self.kfold = None


    def print_epoch(self, epoch, avg_train_epoch_loss,
                    strt,
                    avg_val_epoch_loss=None,
                    cur_patience=None):

        print("Epoch {}:".format(epoch+1))
        print("Training Stats...")
        print("Train loss: {} ".format(avg_train_epoch_loss))
        if self.metrics:
            metricts_dict = self.metrics.get_metrics(type='train')
            for key in metricts_dict:
                print("{}: {} ".format(key,metricts_dict[key]))

        if not avg_val_epoch_loss is None:
            print("Validation Stats...")
            print("Val loss: {}".format(avg_val_epoch_loss))
            if self.metrics:
                metricts_dict = self.metrics.get_metrics(type='val')
                for key in metricts_dict:
                    print("{}: {} ".format(key, metricts_dict[key]))
                if self.kfold:
                    self.kfold_results[self.fold] = metricts_dict
            if self.early_stopping:
                print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if self.kfold:
            torch.save(self.model.state_dict(), os.path.join(
                self.checkpoint_dir,
                'model_checkpoint_fold_{}_epoch_{}.pth'.format(self.fold,epoch)))
            torch.save(self.optimizer.state_dict(), os.path.join(
                self.checkpoint_dir, 'optimizer_checkpoint'))
        else:
            torch.save(self.model.state_dict(), os.path.join(
                self.checkpoint_dir, 'model_checkpoint_epoch_{}.pth'.format(epoch)))
            torch.save(self.optimizer.state_dict(), os.path.join(
                self.checkpoint_dir,'optimizer_checkpoint'))
        # we use the proposed method for saving EncoderDecoder model
        # self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))

    def _compute_train_loss(self, batch,return_scores=False):
        """Compute train loss."""
        if return_scores:
            scores,accum_loss, accum_losses = self._compute_loss(batch,
                                                                 return_scores=True)
            return scores,accum_loss,accum_losses
        else:
            accum_loss, accum_losses = self._compute_loss(batch)
            return accum_loss,accum_losses

    @torch.no_grad()
    def _compute_validation_loss(self):
        """Compute validation loss."""
        self.model.eval()
        loss = 0
        print_loss = defaultdict(int)
        for batch in tqdm(self.val_loader):
            scores,loss_val, loss_dict = self._compute_loss(batch,
                                                       return_scores=True)
            loss += loss_val.item()
            for key, val in loss_dict.items():
                print_loss[key] += val.sum().item()
            if self.metrics is not None:
                self.metrics._compute_metrics_batched(scores,batch[1],
                                                      type='val')

        loss /= len(self.val_loader)
        for key in print_loss:
            print_loss[key] /= len(self.val_loader)
        self.model.train()
        return loss, print_loss

    def _compute_loss(self, batch,return_scores=False):
        """
        To be implemented in wrapper (overwrite)!!
        """
        pass

    def train_epoch(self, epoch):

        self.model.train()
        keep_training = True
        strt = time.time()
        losses = defaultdict(float)
        losses["Total"]=0
        for index, sample_batch in enumerate(tqdm(self.train_loader)):
            scores, loss, t_losses = self._compute_train_loss(sample_batch,
                                                              return_scores=True)
            if self.metrics is not None:
                self.metrics._compute_metrics_batched(scores,sample_batch[
                    1], type='train')
            losses["Total"] +=loss.item()
            loss.backward(retain_graph=True)
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.clip)
            self.optimizer.step()
            self._steps += 1
            if self._steps % 10 == 0:
                self.writer.add_scalars('Train Loss', t_losses,
                                        self._train_logs)
                self._train_logs += 1

        if not self.skip_val and self.val_loader is not None:
            val_loss, val_losses = self._compute_validation_loss()
            self.writer.add_scalars('Val Loss', val_losses, self._val_logs)
            self._val_logs += 1

            if self.early_stopping:
                if val_loss < self.best_val_loss:
                    self.save_epoch(epoch)
                    self.best_val_loss = val_loss
                    self.cur_patience = 0
                    if self.kfold and self.metrics:
                        self.kfold_results[
                            self.fold] = self.metrics.get_metrics(type='val')
                else:
                    self.cur_patience += 1

                if self.cur_patience == self.patience:
                    keep_training = False
        if not self.skip_val and self.val_loader is not None:
            self.print_epoch(epoch,
                             losses["Total"]/len(self.train_loader),
                             strt,
                             avg_val_epoch_loss=val_loss,
                             cur_patience=self.cur_patience)
        else:
            self.print_epoch(epoch,
                             losses["Total"]/len(self.train_loader),
                             strt)
        return keep_training

    def fit(self, epochs, train_loader, val_loader=None):
        self.cur_patience = 0
        self.best_val_loss = 100000000.0
        self.train_loader = train_loader
        self.val_loader = val_loader

        for epoch in range(epochs):
            keep_training = self.train_epoch(epoch)
            if self.metrics:
                self.metrics.reset()
            if not keep_training:
                print("Model converged. Exit...")
                break

    def run_fold(self,fold,epochs,dataset,train_ids,test_ids,batch_size,
                 collator_fn):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_subsampler,
            collate_fn=collator_fn)
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_subsampler,
            collate_fn=collator_fn)

        print("FOLD: ",fold)
        self.fold=fold
        self.fit(epochs=epochs, train_loader=trainloader,
                 val_loader=testloader)
        if self.pretrained:
            state_dict = torch.load(self.pret_dir, map_location='cpu')
            self.model.load_state_dict(state_dict)
        else:
            self.model.apply(reset_weights)

    def kfoldvalidation(self, k_folds, epochs, dataset, batch_size,
                        collator_fn,pretr_dir=None):
        """
        This function is used for kfoldvalidation
        :param k_folds: number of folds
        :param epochs: number of epochs to be executed for each fold
        :param dataset: dataset to be splitted into folds
        :param batch_size: batch size for each iteration
        :param collator_fn: collator function used for dataloader
        """
        kfold = KFold(n_splits=k_folds, shuffle=True)
        self.kfold=True
        self.kfold_results={}
        if pretr_dir is None:
            self.pretrained = False
        else:
            self.pretrained = True
            self.pret_dir = pretr_dir
        if self.pretrained:
            torch.save(self.model.state_dict(), os.path.join(
                self.pret_dir,
                'model_checkpoint_pretrained.pth'))

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            self.run_fold(fold,epochs,dataset, train_ids,test_ids,
                          batch_size, collator_fn)
        if self.metrics:
            metric_keys = self.kfold_results[0].keys()
            for key in metric_keys:
                print("Average {} over folds : ".format(key),sum ([
                    self.kfold_results[
                                                           fold][key] for fold
                                                       in range(
                    k_folds)]) / k_folds)
        self.kfold=None
        self.pretrained = False

    def test(self, testloader,loadckpt=None):
        """Used for evaluating the model"""
        if loadckpt:
        #     load model from checkpoint...
            pass
        self.model.eval()
        self.metrics.reset()
        loss = 0
        print_loss = defaultdict(int)
        for batch in tqdm(testloader):
            scores, loss_val, loss_dict = self._compute_loss(batch,
                                                             return_scores=True)
            loss += loss_val.item()
            for key, val in loss_dict.items():
                print_loss[key] += val.sum().item()
            if self.metrics is not None:
                self.metrics._compute_metrics_batched(scores, batch[1],
                                                      type='val')

        loss /= len(testloader)
        for key in print_loss:
            print_loss[key] /= len(testloader)

        if self.metrics:
            metricts_dict = self.metrics.get_metrics(type='val')
            for key in metricts_dict:
                print("{}: {} ".format(key,metricts_dict[key]))


class ClassificationTrainer(BaseTrainer):

    def __init__(self, model,
                 optimizer,
                 config,
                 metrics=None,
                 scheduler=None):
        """Initiliaze train/test instance."""
        super().__init__(model,optimizer,config,metrics,scheduler)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def _compute_loss(self, batch, return_scores=False):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self.model(batch[0])
        targets = batch[1]

        # Losses
        # Loss: CE
        losses = {'CE': self.criterion(outputs, targets)}
        loss = losses['CE']
        if return_scores:
            return outputs, loss, losses
        else:
            return loss, losses

