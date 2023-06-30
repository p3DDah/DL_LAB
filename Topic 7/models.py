import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch.nn.functional as F


class TKGE(nn.Module, ABC):
    def __init__(self, sizes, rank, margin, bias, init_size):
        super(TKGE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.bias = bias
        self.init_size = init_size
        self.margin = torch.Tensor([margin])

    @abstractmethod
    def get_queries(self, x_data, eval_mode):
        pass

    def score(self, lhs, rhs, eval_mode=False):
        lhs_e, lhs_biases = lhs # head, relation, timestamp
        rhs_e, rhs_biases = rhs # tail
        score = -torch.norm(lhs_e - rhs_e, dim=1)
        if self.bias == 'constant':
            return self.margin.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return x_biases + y_biases.t() + score
            else:
                return x_biases + y_biases + score
        else:
            return score

    def forward(self, x_data, eval_mode=False):
        # get embeddings and similarity scores
        lhs_e, rhs_e, lhs_biases, rhs_biases, factors = self.get_queries(x_data)

        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)

        return predictions, factors

class BaseE(TKGE):
    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.margin, args.bias,args.init_size)

    def marginrankingloss(self, positive_score, negative_score):
        if torch.cuda.is_available():
            rl = nn.ReLU().cuda()
            l = torch.sum(rl(negative_score - positive_score))
        else:
            rl = nn.ReLU()
            l = torch.sum(rl(negative_score - positive_score))
        return l

    def loglikelihoodloss(self, positive_score, negative_score):
        positive_sample_loss = -F.logsigmoid(positive_score).mean()
        negative_sample_loss = -F.logsigmoid(-negative_score).mean()
        l = (positive_sample_loss + negative_sample_loss) / 2
        return l

class NaiveTransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    def __init__(self, args):
        super(NaiveTransE, self).__init__(args)

        self.embeddings = nn.ModuleList([])
        self.embeddings.append(nn.Embedding(self.sizes[0],self.rank))
        self.embeddings.append(nn.Embedding(self.sizes[1],self.rank//2))
        self.embeddings.append(nn.Embedding(self.sizes[3],self.rank//2))

        self.bh = nn.Embedding(self.sizes[0], 1)
        self.bh.weight.data = torch.zeros((self.sizes[0], 1))
        self.bt = nn.Embedding(self.sizes[0], 1)
        self.bt.weight.data = torch.zeros((self.sizes[0], 1))

        # entities
        self.embeddings[0].weight.data = self.init_size * torch.randn((self.sizes[0], self.rank))
        # relations: half dimension because of concatenation
        self.embeddings[1].weight.data = self.init_size * torch.randn((self.sizes[0], self.rank//2))
        # timestamps: half dimension because of concatenation
        self.embeddings[2].weight.data = self.init_size * torch.randn((self.sizes[0], self.rank//2))

    def get_queries(self, x_data, eval_mode = False):
        head_e = self.embeddings[0](x_data[:,0])
        rel_e = self.embeddings[1](x_data[:,1])
        ts_e = self.embeddings[2](x_data[:,3])
        lhs_biases = self.bh(x_data[:, 0])

        if eval_mode:
            tail_e, rhs_biases = self.embeddings[0].weight, self.bt.weight
        else:
            tail_e, rhs_biases = self.embeddings[0](x_data[:,2]), self.bt(x_data[:,2])

        lhs = head_e + torch.cat((rel_e, ts_e), dim=-1)
        rhs = tail_e
        # score = -torch.norm(head_e + torch.cat((rel_e,ts_e),dim=-1) - tail_e, dim=1)
        return lhs, rhs, lhs_biases, rhs_biases, [head_e, rel_e, tail_e, ts_e]



class VectorTransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(VectorTransE, self).__init__(args)

        self.embeddings = nn.ModuleList([nn.Embedding(s, self.rank) for s in [sizes[0], sizes[1], sizes[3]]])
        # entities
        self.embeddings[0].weight.data = self.init_size * torch.randn((self.sizes[0], self.rank))
        # relations
        self.embeddings[1].weight.data = self.init_size * torch.randn((self.sizes[0], self.rank))
        # timestamps
        self.embeddings[2].weight.data = self.init_size * torch.randn((self.sizes[0], self.rank))

    def get_queries(self, x_data, eval_mode=False):
        head_e = self.embeddings[0](x_data[:, 0])
        rel_e = self.embeddings[1](x_data[:, 1])
        ts_e = self.embeddings[2](x_data[:, 3])
        x_biases = self.bh(x_data[:, 0])

        if eval_mode:
            tail_e, y_biases = self.embeddings[0].weight, self.bt.weight
        else:
            tail_e, y_biases = self.embeddings[0](x_data[:, 2]), self.bt(x_data[:, 2])

        x = head_e + rel_e + ts_e
        y = tail_e
        # score = -torch.norm(head_e + torch.cat((rel_e,ts_e),dim=-1) - tail_e, dim=1)
        return x, y, x_biases, y_biases, [head_e, rel_e, tail_e, ts_e]