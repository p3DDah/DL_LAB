import collections
import os
import pickle as pkl
import numpy as np


def get_idx(path):
    entities, relations, timestamps = set(), set(), set()
    for split in ["train.txt", "valid.txt", "test.txt"]:
        with open(os.path.join(path, split), "r", encoding="utf8") as lines:
            for line in lines:
                head, rel, tail, temp = line.strip().split("\t")
                entities.add(head)
                relations.add(rel)
                entities.add(tail)
                timestamps.add(temp)

    ent2idx = {i: x for (x, i) in enumerate(sorted(entities))}
    rel2idx = {i: x for (x, i) in enumerate(sorted(relations))}
    ts2idx = {i: x for (x, i) in enumerate(sorted(timestamps))}
    idx2ent = {i: x for (i, x) in enumerate(sorted(entities))}
    idx2rel = {i: x for (i, x) in enumerate(sorted(relations))}
    idx2ts = {i: x for (i, x) in enumerate(sorted(timestamps))}

    return ent2idx, rel2idx, ts2idx, idx2ent, idx2rel, idx2ts


def to_np_array(dataset_file, ent2idx, rel2idx, ts2idx):
    examples = []
    with open(dataset_file, "r", encoding="utf8") as lines:
        for line in lines:
            lhs, rel, rhs, temp = line.strip().split("\t")
            try:
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs], ts2idx[temp]])
            except KeyError:
                print("KeyError", lhs, rel, rhs, temp)
                continue
    return np.array(examples).astype("int64")

def get_filters(examples, n_relations, n_timestamps):
    head_filters = collections.defaultdict(set)
    tail_filters = collections.defaultdict(set)
    for lhs, rel, rhs, temp in examples:
        tail_filters[(lhs, rel, temp)].add(rhs)
        head_filters[(rhs, rel + n_relations, temp+n_timestamps)].add(lhs)
    head_final = {}
    tail_final = {}
    for k, v in head_filters.items():
        head_final[k] = sorted(list(v))
    for k, v in tail_filters.items():
        tail_final[k] = sorted(list(v))
    return head_final, tail_final

def process_dataset(path):
    ent2idx, rel2idx, ts2idx, idx2ent, idx2rel, idx2ts = get_idx(path)
    examples = {}
    splits = ["train.txt", "valid.txt", "test.txt"]
    for split in splits:
        dataset_file = os.path.join(path, split)
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx, ts2idx)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    head_skip, tail_skip = get_filters(all_examples, len(rel2idx), len(ts2idx))
    filters = {"head": head_skip, "tail": tail_skip}

    return ent2idx, rel2idx, ts2idx, idx2ent, idx2rel, idx2ts, examples, filters


def ds_read(data_path):
    for dataset_name in os.listdir(data_path):
        dataset_path = os.path.join(data_path, dataset_name)

        ent2idx, rel2idx, ts2idx, idx2ent, idx2rel, idx2ts, dataset_examples, dataset_filters = process_dataset(dataset_path)

        for dataset_split in ["train.txt", "valid.txt", "test.txt"]:
            save_path = os.path.join(dataset_path, dataset_split + ".pickle")
            with open(save_path, "wb") as save_file:
                pkl.dump(dataset_examples[dataset_split], save_file)

        with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
            pkl.dump(dataset_filters, save_file)

        save_path = os.path.join(dataset_path, 'entities' + ".dict")
        with open(save_path, "wb") as save_file:
            pkl.dump(ent2idx, save_file)
        save_path = os.path.join(dataset_path, 'relations' + ".dict")
        with open(save_path, "wb") as save_file:
            pkl.dump(rel2idx, save_file)
        save_path = os.path.join(dataset_path, 'timestamps' + ".dict")
        with open(save_path, "wb") as save_file:
            pkl.dump(ts2idx, save_file)

        save_path = os.path.join(dataset_path, 'idx2entity' + ".dict")
        with open(save_path, "wb") as save_file:
            pkl.dump(idx2ent, save_file)
        save_path = os.path.join(dataset_path, 'idx2relation' + ".dict")
        with open(save_path, "wb") as save_file:
            pkl.dump(idx2rel, save_file)
        save_path = os.path.join(dataset_path, 'idx2timestamps' + ".dict")
        with open(save_path, "wb") as save_file:
            pkl.dump(idx2ts, save_file)