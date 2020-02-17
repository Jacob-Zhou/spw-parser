# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import cky, crf
from parser.utils.common import bos, eos, pad, unk
from parser.utils.corpus import Corpus, Treebank
from parser.utils.field import (BertField, ChartField, Field,
                                RawField)
from parser.utils.fn import build, factorize
from parser.utils.metric import BracketMetric

import torch
import torch.nn as nn
from transformers import BertTokenizer


class CMD(object):

    def __call__(self, args):
        self.args = args
        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")
            self.TREE = RawField('trees')
            self.CHAR = Field('chars', pad=pad, unk=unk,
                              bos=bos, eos=eos, lower=True)
            self.POS = Field('pos')

            self.FEAT = None
            self.CHART = ChartField('charts')
            if args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert',
                                      pad='[PAD]',
                                      bos='[CLS]',
                                      eos='[SEP]',
                                      tokenize=tokenizer.encode)
                self.fields = Treebank(TREE=self.TREE,
                                       CHAR=(self.CHAR, self.FEAT),
                                       POS=self.POS,
                                       CHART=self.CHART)
            else:
                self.fields = Treebank(TREE=self.TREE,
                                       CHAR=self.CHAR,
                                       POS=self.POS,
                                       CHART=self.CHART)

            train = Corpus.load(args.ftrain, self.fields)
            if args.fembed:
                embed = Embedding.load(args.fembed, args.unk)
            else:
                embed = None
            self.CHAR.build(train, args.min_freq, embed)
            if self.FEAT:
                self.FEAT.build(train)
            self.CHART.build(train)
            self.POS.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            self.TREE = self.fields.TREE
            self.CHAR = self.fields.CHAR
            self.POS = self.fields.POS
            self.CHART = self.fields.CHART
        self.criterion = nn.CrossEntropyLoss()

        args.update({
            'n_chars': self.CHAR.vocab.n_init,
            'n_labels': len(self.CHART.vocab),
            'n_pos_labels': len(self.POS.vocab),
            'pad_index': self.CHAR.pad_index,
            'unk_index': self.CHAR.unk_index,
            'bos_index': self.CHAR.bos_index,
            'eos_index': self.CHAR.eos_index
        })

        print(f"Override the default configs\n{args}")
        print(f"{self.TREE}\n{self.CHAR}\n{self.POS}\n{self.CHART}")

    def train(self, loader):
        self.dp_model.train()

        for trees, chars, (spans, labels) in loader:
            self.optimizer.zero_grad()

            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            span_mask = spans.gt(0)
            spans = torch.nn.functional.one_hot(spans, 5).bool()[..., 1:]
            s_span, s_label, span_loss = self.dp_model(
                chars, None, mask, spans)
            span_mask = span_mask & mask
            label_loss = self.criterion(s_label[span_mask], labels[span_mask])
            loss = span_loss.mean() + label_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.dp_model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()
        # exit()

    @torch.no_grad()
    def evaluate(self, loader):
        self.dp_model.eval()

        total_loss = 0
        metric = BracketMetric(self.POS.vocab.stoi.keys())

        for trees, chars, (spans, labels) in loader:
            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            span_mask = spans.gt(0)
            spans = torch.nn.functional.one_hot(spans, 5).bool()[..., 1:]
            s_span, s_label, span_loss = self.dp_model(
                chars, None, mask, spans)
            span_mask = span_mask & mask
            loss = span_loss.mean() + \
                self.criterion(s_label[span_mask], labels[span_mask])
            preds = self.model.decode(s_span, s_label, mask)
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            total_loss += loss.item()
            metric([factorize(tree, self.args.delete, self.args.equal)
                    for tree in preds],
                   [factorize(tree, self.args.delete, self.args.equal)
                    for tree in trees])
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.dp_model.eval()

        all_trees = []
        for trees, chars in loader:
            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            s_span, s_label, _ = self.dp_model(chars, None, mask, None)
            preds = self.model.decode(s_span, s_label, mask)
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            all_trees.extend(preds)

        return all_trees
