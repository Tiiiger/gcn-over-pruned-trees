"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import pickle as pkl

from utils import constant, helper, vocab
from model.tree import head_to_tree, tree_to_adj

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False,
                 phase="train"):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        try:
            with open("./dataset/preprocessed/data.{}.{}.{}.pkl".format(phase, opt["graph_model"], opt["num_layers"]), "rb") as f:
                data = pkl.load(f)
        except:
            data = self.preprocess(data, vocab, opt)
            with open("./dataset/preprocessed/data.{}.{}.{}.pkl".format(phase, opt["graph_model"], opt["num_layers"]), "wb") as f:
                pkl.dump(data, f)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-2]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def input_to_adj(self, head, words, prune_k, subj_pos, obj_pos):
        tree = head_to_tree(head, words, len(words), prune_k, subj_pos, obj_pos)
        adj = tree_to_adj(len(words), tree, directed=False, self_loop=False)
        adj = torch.from_numpy(adj)
        return adj

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            # tokens = torch.LongTensor(tokens)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            pos = torch.LongTensor(pos)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            ner = torch.LongTensor(ner)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            deprel = torch.LongTensor(deprel)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            subj_positions = torch.LongTensor(subj_positions)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            obj_positions = torch.LongTensor(obj_positions)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            subj_type = torch.LongTensor(subj_type)
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            obj_type = torch.LongTensor(obj_type)
            relation = self.label2id[d['relation']]
            adj = self.input_to_adj(head, tokens, opt["prune_k"], subj_positions, obj_positions)
            adj_mask = (adj.sum(0)+adj.sum(1)).eq(0)
            adj = adj + torch.eye(adj.size(0))
            adj /= adj.sum(1, keepdim=True)
            if opt['graph_model'] == 'SGR':
                for _ in range(opt['num_layers']-1):
                    adj = torch.mm(adj, adj)
            processed += [(tokens, pos, ner, deprel, adj, subj_positions, obj_positions, subj_type, obj_type, relation, adj_mask)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 11

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        max_len = len(words[0])
        words = get_long_tensor(words, batch_size, max_len=max_len)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size, max_len=max_len)
        ner = get_long_tensor(batch[2], batch_size, max_len=max_len)
        deprel = get_long_tensor(batch[3], batch_size, max_len=max_len)
        adjs = torch.zeros(batch_size, words.size(1), words.size(1))
        for i, adj in enumerate(batch[4]):
            adjs[i, :adj.size(0), :adj.size(1)] = adj
        adj_masks = torch.ones(batch_size, words.size(1)).byte()
        for i, adj_mask in enumerate(batch[10]):
            adj_masks[i, :adj_mask.size(0)] = adj_mask
        adj_masks = adj_masks.unsqueeze(2)
        # old_adj_masks = (adjs.sum(2) + adjs.sum(1)).eq(0).unsqueeze(2)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])

        return (words, masks, pos, ner, deprel, adjs, adj_masks, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size, max_len=0):
    """ Convert list of list of tokens to a padded LongTensor. """
    if max_len==0:
        token_len = max(len(x) for x in tokens_list)
    else:
        token_len=max_len
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        if isinstance(s, list):
            tokens[i, :len(s)] = torch.LongTensor(s)
        else:
            tokens[i, :s.size(0)] = s
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

