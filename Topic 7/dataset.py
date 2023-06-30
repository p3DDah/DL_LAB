import collections
import os
import pickle as pkl
import numpy as np
import torch

class TKGEDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, file_name, debug):
        # Base KG with entity and relation lists
        self.data_path = data_path
        self.debug = debug
        self.data = {}

        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        filters_file.close()

        file_path = os.path.join(self.data_path, 'entities' + ".dict")
        with open(file_path, "rb") as in_file:
            self.entity2id = pkl.load(in_file)

        file_path = os.path.join(self.data_path, 'relations' + ".dict")
        with open(file_path, "rb") as in_file:
            self.relation2id = pkl.load(in_file)

        file_path = os.path.join(self.data_path, 'timestamps' + ".dict")
        with open(file_path, "rb") as in_file:
            self.timestamp2id = pkl.load(in_file)

        file_path = os.path.join(self.data_path, file_name)
        with open(file_path, "rb") as in_file:
            self.trainquadruples = pkl.load(in_file)

        self.lenhead = len(self.trainquadruples[:, 0])
        self.lenrel = len(self.trainquadruples[:, 1])
        self.lentail = len(self.trainquadruples[:, 2])
        self.lents = len(self.trainquadruples[:, 3])

    def __getitem__(self, index):
        samples = self.trainquadruples[index, :]
        headn, relationn, tailn, timestampn = samples[:, 0], \
            samples[:, 1], samples[:, 2], samples[:, 3]

        # From numpy to tensor ...
        if torch.cuda.is_available():
            head = torch.cuda.LongTensor(headn)
            relation = torch.cuda.LongTensor(relationn)
            tail = torch.cuda.LongTensor(tailn)
            timestamp = torch.cuda.LongTensor(timestampn)
        else:
            head = torch.LongTensor(headn)
            relation = torch.LongTensor(relationn)
            tail = torch.LongTensor(tailn)
            timestamp = torch.LongTensor(timestampn)
        return torch.stack([head, relation, tail, timestamp])

    def get_shape(self):
        return self.lenhead, self.lenrel, self.lentail, self.lents

    def merge_and_create_dic(self, train_ds, val_ds, test_ds):
        all_quadruple_rank_head = torch.cat((train_ds[0], val_ds[0], test_ds[0]), dim=0)
        all_quadruple_rank_rel = torch.cat((train_ds[1], val_ds[1], test_ds[1]), dim=0)
        all_quadruple_rank_tail = torch.cat((train_ds[2], val_ds[2], test_ds[2]), dim=0)
        all_quadruple_rank_ts = torch.cat((train_ds[3], val_ds[3], test_ds[3]), dim=0)

        all_quadruple_rank_h = dict()
        all_quadruple_rank_t = dict()
        all_quadruple_rank_raw_h = dict()
        all_quadruple_rank_raw_t = dict()

        for i in range(len(all_quadruple_rank_head)):
            key = (all_quadruple_rank_head[i].item(), all_quadruple_rank_rel[i].item(), all_quadruple_rank_ts[i].item())
            if key not in all_quadruple_rank_t:
                all_quadruple_rank_t[key] = []
                # append some value
            all_quadruple_rank_t[key].append(all_quadruple_rank_tail[i])

            key = (all_quadruple_rank_rel[i].item(), all_quadruple_rank_tail[i].item(), all_quadruple_rank_ts[i].item())
            if key not in all_quadruple_rank_h:
                all_quadruple_rank_h[key] = []
                # append some value
            all_quadruple_rank_h[key].append(all_quadruple_rank_head[i])

            key = (all_quadruple_rank_head[i].item(), all_quadruple_rank_rel[i].item())
            if key not in all_quadruple_rank_raw_t:
                all_quadruple_rank_raw_t[key] = []
                # append some value
            all_quadruple_rank_raw_t[key].append(all_quadruple_rank_tail[i])

            key = (all_quadruple_rank_rel[i].item(), all_quadruple_rank_tail[i].item())
            if key not in all_quadruple_rank_raw_h:
                all_quadruple_rank_raw_h[key] = []
                # append some value
            all_quadruple_rank_raw_h[key].append(all_quadruple_rank_head[i])


        return all_quadruple_rank_t, all_quadruple_rank_h, all_quadruple_rank_raw_t, all_quadruple_rank_raw_h



