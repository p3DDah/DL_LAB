import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm

class TKGE_Optimizer(object):
    def __init__(
            self, model, regularizer, optimizer, batch_size, neg_sample_size, device, args):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.negative_sample_size = neg_sample_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.n_entities = model.sizes[0]
        self.device = device
        self.args = args

    def reduce_lr(self, factor=0.8):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor

    def negative_sample_generator(self, head):
        negative_samplen = np.random.randint(self.lentr, size=len(head) * self.negative_sample_size)
        return negative_samplen

    def get_neg_samples(self, data, mode= "head"):

        negative_data = data.repeat(self.negative_sample_size, 1)
        data_size = data.shape[0]

        # neg tail
        if mode == "tail":
            negsamples = torch.Tensor(np.random.randint(
                self.n_entities,
                size=data_size * self.negative_sample_size)
            ).to(self.device)
            negative_data[:, 2] = negsamples

        # neg head
        if mode == "head":
            negsamples = torch.Tensor(np.random.randint(
                self.n_entities,
                size=data_size * self.negative_sample_size)
            ).to(self.device)
            negative_data[:, 0] = negsamples

        return negative_data

    def neg_sampling_loss(self, input_batch):
        # positive samples
        positive_score, factors = self.model(input_batch)

        # negative samples head
        neg_samples_head = self.get_neg_samples(input_batch, mode="head")
        negative_score_head, _ = self.model(neg_samples_head)

        # negative samples head
        neg_samples_tail = self.get_neg_samples(input_batch, mode="tail")
        negative_score_tail, _ = self.model(neg_samples_tail)

        # calculate loglikelihoodloss
        #loss = - torch.cat([positive_score, negative_score_head], dim=0).mean()
        #loss += - torch.cat([positive_score, negative_score_tail], dim=0).mean()

        loss = self.model.loglikelihoodloss(positive_score, negative_score_head)/2
        loss += self.model.loglikelihoodloss(positive_score, negative_score_tail)/2

        # loss = self.model.marginrankingloss(positive_score.repeat(self.args.neg_sample_size), negative_score_head) / len(negative_score_head)
        # loss += self.model.marginrankingloss(positive_score.repeat(self.args.neg_sample_size), negative_score_tail) / len(negative_score_tail)

        return loss, factors

    def calculate_loss(self, input_batch):
        if self.negative_sample_size > 0:
            loss, factors = self.neg_sampling_loss(input_batch)
        else:
            predictions, factors = self.model(input_batch, eval_mode=True)
            truth_tail = input_batch[:, 2]
            truth_head = input_batch[:, 0]
            loss = self.loss_fn(predictions, truth_tail)
            loss += self.loss_fn(predictions, truth_head)

        # regularization loss
        loss += self.regularizer.forward(factors)
        return loss

    def calculate_valid_loss(self, examples):
        train_loader = torch.utils.data.DataLoader(
            examples, batch_size=self.batch_size, shuffle=True)

        totalloss = 0.0

        with torch.no_grad():
            for input_batch in train_loader:
                loss = self.calculate_loss(input_batch)
                totalloss += loss

        mean_loss = totalloss.item()/len(train_loader)
        return totalloss, mean_loss

    def epoch(self, examples, step, max_epochs):
        train_loader = torch.utils.data.DataLoader(
            examples, batch_size=self.batch_size, shuffle=True)
        totalloss = 0

        with tqdm.tqdm(total=len(train_loader), unit='exec') as bar:
            bar.set_description(f'\tTraining progress')
            for input_batch in train_loader:
                # gradient step
                loss = self.calculate_loss(input_batch)
                totalloss += loss

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                bar.update(1)
                bar.set_postfix(epoch=f'{step + 1}/{max_epochs}', loss=f'{loss:.4f}')
            bar.close()

        mean_loss = totalloss.item()/len(train_loader)

        return totalloss.detach().cpu(), mean_loss