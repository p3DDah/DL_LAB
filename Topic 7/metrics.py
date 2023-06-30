'''
Created on May 21, 2023

@author: Mojtaba Nayyeri
'''

from _operator import truediv
from config import device
import torch
import numpy as np
import random
import tqdm
import os


#expecting 1D np array of predictions and labels.
def ranking(x, model,all_quadruple_rank_t, all_quadruple_rank_h, filter_rank_raw_t, filter_rank_raw_h,batches):

    head, relation, tail, timestamp= x[:,0], x[:,1], x[:,2], x[:,3]
    length = len(head)
    num_batches = batches
    batch_size = length//num_batches

    all_entity = torch.LongTensor(np.arange(0, length)).to(device)
    head_expand = torch.ones(length).to(device)
    tail_expand = torch.ones(length).to(device)
    relation_expand = torch.ones(length).to(device)
    timestamp_expand = torch.ones(length).to(device)

    total_rank = 0
    mr = 0
    mrr = 0
    hits10 = 0
    hits3 = 0
    hits1 = 0

    total_rank_raw = 0
    mr_raw = 0
    mrr_raw = 0
    hits10_raw = 0
    hits3_raw = 0
    hits1_raw = 0

    with torch.no_grad():
        with tqdm.tqdm(total=batch_size, unit='exec') as bar:
            bar.set_description(f'\tEvaluation progress')
            for idx in range(batch_size):
                h, r, t, ts = head[idx] * head_expand, \
                    relation[idx] * relation_expand, \
                    tail[idx] * tail_expand, \
                    timestamp[idx] * timestamp_expand,
                h, r, t, ts = h.type(torch.LongTensor).to(device), r.type(torch.LongTensor).to(device), \
                    t.type(torch.LongTensor).to(device), ts.type(torch.LongTensor).to(device)

                filter_rank_t = all_quadruple_rank_t
                filter_rank_h = all_quadruple_rank_h


                filter_tail = filter_rank_t[(head[idx].item(), relation[idx].item(), timestamp[idx].item())]
                filter_head = filter_rank_h[(relation[idx].item(), tail[idx].item(), timestamp[idx].item())]

                filter_tail_raw = filter_rank_raw_t[(head[idx].item(), relation[idx].item())]
                filter_head_raw = filter_rank_raw_h[(relation[idx].item(), tail[idx].item())]


                x = torch.stack((torch.tensor([head[idx]]).to(device), torch.tensor([relation[idx]]).to(device), torch.tensor(
                   [tail[idx]]).to(device), torch.tensor([timestamp[idx]]).to(device)), dim=1)

                Corrupted_score_tail, _ = model(torch.stack((h, r, all_entity, ts), dim=1))
                Corrupted_score_tail_raw = Corrupted_score_tail.clone()

                indices = torch.cat([tensor.flatten() for tensor in filter_tail])
                indices_raw = torch.cat([tensor.flatten() for tensor in filter_tail_raw])
                mask = torch.ones_like(Corrupted_score_tail).bool()
                mask[indices] = False
                Corrupted_score_tail[~mask] = -float('inf')
                mask[indices_raw] = False
                Corrupted_score_tail_raw[~mask] = -float('inf')


                Corrupted_score_head, _ = model(torch.stack((all_entity, r, t, ts), dim=1))
                Corrupted_score_head_raw = Corrupted_score_head.clone()

                indices = torch.cat([tensor.flatten() for tensor in filter_head])
                mask = torch.ones_like(Corrupted_score_head).bool()
                mask[indices] = False
                Corrupted_score_head[~mask] = -float('inf')
                mask[indices_raw] = False
                Corrupted_score_head_raw[~mask] = -float('inf')
                Quadruple, _ = model(x)
                #Quadruple = torch.sub(Quadruple, 1e-5)

                ranking_tail = torch.sum((torch.gt(Corrupted_score_tail, Quadruple)).float()) + 1
                ranking_head = torch.sum((torch.gt(Corrupted_score_head, Quadruple)).float()) + 1

                ranking_tail_raw = torch.sum((Corrupted_score_tail_raw > Quadruple).float()) + 1
                ranking_head_raw = torch.sum((Corrupted_score_head_raw > Quadruple).float()) + 1

                avg_rank = (1/ranking_head + 1/ranking_tail) / 2
                mr += (ranking_head + ranking_tail) / 2
                total_rank += avg_rank
                hits10 += 1 if (ranking_head + ranking_tail) / 2 < 11 else 0
                hits3 += 1 if (ranking_head + ranking_tail) / 2 < 4 else 0
                hits1 += 1 if (ranking_head + ranking_tail) / 2 < 2 else 0

                avg_rank_raw = (1 / ranking_head_raw + 1 / ranking_tail_raw) / 2
                mr_raw += (ranking_head_raw + ranking_tail_raw) / 2
                total_rank_raw += avg_rank_raw
                hits10_raw += 1 if (ranking_head_raw + ranking_tail_raw) / 2 < 11 else 0
                hits3_raw += 1 if (ranking_head_raw + ranking_tail_raw) / 2 < 4 else 0
                hits1_raw += 1 if (ranking_head_raw + ranking_tail_raw) / 2 < 2 else 0
                bar.update(1)
                bar.set_postfix(Hits10=f'{hits10}/{batch_size}',
                                Hits3=f'{hits3}/{batch_size}',
                                Hits1=f'{hits1}/{batch_size}',
                                MRR=f'{avg_rank.item():.6f}',
                                MR=f'{(ranking_head + ranking_tail) / 2:.6f}')
            bar.close()

    mr = mr / batch_size
    mrr = total_rank / batch_size
    tmp_hits10, tmp_hits3, tmp_hits1 = hits10, hits3, hits1
    hits10 = hits10 / batch_size
    hits3 = hits3 / batch_size
    hits1 = hits1 / batch_size

    mr_raw = mr_raw / batch_size
    mrr_raw = total_rank_raw / batch_size
    tmp_hits10_raw, tmp_hits3_raw, tmp_hits1_raw = hits10_raw, hits3_raw, hits1_raw
    hits10_raw = hits10_raw / batch_size
    hits3_raw = hits3_raw / batch_size
    hits1_raw = hits1_raw / batch_size

    return {"MR": mr, "MRR":mrr,"HITS10": hits10,"HITS3": hits3,"HITS1": hits1,"MR_RAW": mr_raw, "MRR_RAW": mrr_raw,"HITS10_RAW": hits10_raw, "HITS3_RAW": hits3_raw,"HITS1_RAW": hits1_raw}


