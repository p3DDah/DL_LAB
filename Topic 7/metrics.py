from _operator import truediv
from config import device
import torch
import numpy as np
import random
import tqdm
import os


#expecting 1D np array of predictions and labels.
def ranking(x, model,filter_rank_t, filter_rank_h, filter_rank_raw_t, filter_rank_raw_h,batches):

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

    total_rank_xxx = 0
    mr_xxx = 0
    mrr_xxx = 0
    hits10_xxx = 0
    hits3_xxx = 0
    hits1_xxx = 0

    total_rank_xx = 0
    mr_xx = 0
    mrr_xx = 0
    hits10_xx = 0
    hits3_xx = 0
    hits1_xx = 0

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


                filter_tail_hrt = filter_rank_t[(head[idx].item(), relation[idx].item(), timestamp[idx].item())]
                filter_head_trt= filter_rank_h[(relation[idx].item(), tail[idx].item(), timestamp[idx].item())]

                filter_tail_hr = filter_rank_raw_t[(head[idx].item(), relation[idx].item())]
                filter_head_tr = filter_rank_raw_h[(relation[idx].item(), tail[idx].item())]


                x = torch.stack((torch.tensor([head[idx]]).to(device), torch.tensor([relation[idx]]).to(device), torch.tensor(
                   [tail[idx]]).to(device), torch.tensor([timestamp[idx]]).to(device)), dim=1)

                Corrupted_score_tail, _ = model(torch.stack((h, r, all_entity, ts), dim=1))
                Corrupted_score_tail_hrt = Corrupted_score_tail.clone()
                Corrupted_score_tail_hr = Corrupted_score_tail.clone()

                indices_hrt = torch.cat([tensor.flatten() for tensor in filter_tail_hrt])
                indices_hr = torch.cat([tensor.flatten() for tensor in filter_tail_hr])
                mask = torch.ones_like(Corrupted_score_tail).bool()
                mask[indices_hrt] = False
                mask[indices_hr] = False
                Corrupted_score_tail_hrt[~mask] = -float('inf')
                Corrupted_score_tail_hr[~mask] = -float('inf')


                Corrupted_score_head, _ = model(torch.stack((all_entity, r, t, ts), dim=1))
                Corrupted_score_head_trt = Corrupted_score_head.clone()
                Corrupted_score_head_tr = Corrupted_score_head.clone()

                indices_trt = torch.cat([tensor.flatten() for tensor in filter_head_trt])
                indices_tr = torch.cat([tensor.flatten() for tensor in filter_head_tr])
                mask = torch.ones_like(Corrupted_score_head).bool()
                mask[indices_trt] = False
                mask[indices_tr] = False
                Corrupted_score_head_trt[~mask] = -float('inf')
                Corrupted_score_head_tr[~mask] = -float('inf')

                Quadruple, _ = model(x)

                ranking_tail = torch.sum((torch.gt(Corrupted_score_tail, Quadruple)).float()) + 1
                ranking_head = torch.sum((torch.gt(Corrupted_score_head, Quadruple)).float()) + 1

                ranking_tail_hrt = torch.sum((Corrupted_score_tail_hrt > Quadruple).float()) + 1
                ranking_head_trt = torch.sum((Corrupted_score_head_trt > Quadruple).float()) + 1

                ranking_tail_hr = torch.sum((Corrupted_score_tail_hr > Quadruple).float()) + 1
                ranking_head_tr = torch.sum((Corrupted_score_head_tr > Quadruple).float()) + 1

                avg_rank = (1/ranking_head + 1/ranking_tail) / 2
                mr += (ranking_head + ranking_tail) / 2
                total_rank += avg_rank
                hits10 += 1 if (ranking_head + ranking_tail) / 2 < 11 else 0
                hits3 += 1 if (ranking_head + ranking_tail) / 2 < 4 else 0
                hits1 += 1 if (ranking_head + ranking_tail) / 2 < 2 else 0

                avg_rank_xxx = (1 / ranking_head_trt + 1 / ranking_tail_hrt) / 2
                mr_xxx += (ranking_head_trt + ranking_tail_hrt) / 2
                total_rank_xxx += avg_rank_xxx
                hits10_xxx += 1 if (ranking_head_trt + ranking_tail_hrt) / 2 < 11 else 0
                hits3_xxx += 1 if (ranking_head_trt + ranking_tail_hrt) / 2 < 4 else 0
                hits1_xxx += 1 if (ranking_head_trt + ranking_tail_hrt) / 2 < 2 else 0

                avg_rank_xx = (1 / ranking_head_tr + 1 / ranking_tail_hr) / 2
                mr_xx += (ranking_head_tr + ranking_tail_hr) / 2
                total_rank_xx += avg_rank_xx
                hits10_xx += 1 if (ranking_head_tr + ranking_tail_hr) / 2 < 11 else 0
                hits3_xx += 1 if (ranking_head_tr + ranking_tail_hr) / 2 < 4 else 0
                hits1_xx += 1 if (ranking_head_tr + ranking_tail_hr) / 2 < 2 else 0

                bar.update(1)
                bar.set_postfix(Hits10=f'{hits10_xx}/{batch_size}',
                                Hits3=f'{hits3_xx}/{batch_size}',
                                Hits1=f'{hits1_xx}/{batch_size}',
                                MRR=f'{avg_rank_xx.item():.6f}',
                                MR=f'{(ranking_head + ranking_tail) / 2:.6f}')
            bar.close()

    mr = mr / batch_size
    mrr = total_rank / batch_size
    tmp_hits10, tmp_hits3, tmp_hits1 = hits10, hits3, hits1
    hits10 = hits10 / batch_size
    hits3 = hits3 / batch_size
    hits1 = hits1 / batch_size

    mr_xxx = mr_xxx / batch_size
    mrr_xxx = total_rank_xxx / batch_size
    tmp_hits10_xxx, tmp_hits3_xxx, tmp_hits1_xxx = hits10_xxx, hits3_xxx, hits1_xxx
    hits10_xxx = hits10_xxx / batch_size
    hits3_xxx = hits3_xxx / batch_size
    hits1_xxx = hits1_xxx / batch_size

    mr_xx = mr_xx / batch_size
    mrr_xx = total_rank_xx / batch_size
    tmp_hits10_xx, tmp_hits3_xx, tmp_hits1_xx = hits10_xx, hits3_xx, hits1_xx
    hits10_xx = hits10_xx / batch_size
    hits3_xx = hits3_xx / batch_size
    hits1_xx = hits1_xx / batch_size

    return {"MR": mr, "MRR":mrr,"HITS10": hits10,"HITS3": hits3,"HITS1": hits1,
            "MR_XXX": mr_xxx, "MRR_XXX": mrr_xxx,"HITS10_XXX": hits10_xxx, "HITS3_XXX": hits3_xxx,"HITS1_XXX": hits1_xxx,
            "MR_XX": mr_xx, "MRR_XX": mrr_xx,"HITS10_XX": hits10_xx, "HITS3_XX": hits3_xx,"HITS1_XX": hits1_xx}


