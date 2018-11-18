"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['hidden_dim']*3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, adjs, adj_masks, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        h = self.gcn(words, masks, pos, ner, adjs)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, adj_masks, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])

        # gcn layer
        layers = []
        if opt['graph_model'] == 'GCN':
            for layer in range(self.layers):
                input_dim = self.in_dim if layer == 0 else self.mem_dim
                if layer != self.layers-1:
                    drop = opt['gcn_dropout']
                else:
                    drop = 0
                layers.append(GCN_layer(input_dim, self.mem_dim, drop))
                layers.append(nn.Dropout(p=drop))
        elif opt['graph_model'] == 'MLP':
            layers.append(MLP(self.in_dim, self.mem_dim, self.mem_dim, opt['gcn_dropout']))
        else:
            layers.append(SGC(self.in_dim, self.mem_dim))

        self.gcn_layers = nn.ModuleList(layers)

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, words, masks, pos, ner, adjs):
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        else:
            gcn_inputs = embs

        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adjs = torch.zeros_like(adjs)

        for l in self.gcn_layers:
            if isinstance(l, nn.Dropout):
                gcn_inputs = l(gcn_inputs)
            else:
                gcn_inputs = l(gcn_inputs, adjs)

        return gcn_inputs

class GCN_layer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.transform = nn.Linear(in_features, out_features)
        self.dropout = dropout
        torch.nn.init.kaiming_normal_(self.transform.weight)

    def forward(self, inputs, adjs):
        AxW = self.transform(adjs.bmm(inputs))
        gAxW = F.relu(AxW)
        gAxW = F.dropout(gAxW, p=self.dropout, inplace=True, training=self.training)
        return gAxW

class SGC(nn.Module):
    def __init__(self, in_features, out_features):
        super(SGC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.transform = nn.Linear(in_features, out_features)
        torch.nn.init.kaiming_normal_(self.transform.weight)

    def forward(self, inputs, adjs):
        # return self.transform(adjs.bmm(inputs))
        return F.relu(self.transform(adjs.bmm(inputs)))

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.transform1 = nn.Linear(in_features, hidden_features)
        self.transform2 = nn.Linear(hidden_features, out_features)
        self.dropout = dropout
        torch.nn.init.kaiming_normal_(self.transform1.weight)
        torch.nn.init.kaiming_normal_(self.transform2.weight)

    def forward(self, inputs, adjs):
        AxW = F.relu(self.transform1(inputs))
        AxW = F.dropout(AxW, p=self.dropout, inplace=True, training=self.training)
        AxW = self.transform2(AxW)
        AxW = adjs.bmm(AxW)
        AxW = F.relu(AxW)
        return AxW

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
