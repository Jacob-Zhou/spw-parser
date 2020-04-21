# -*- coding: utf-8 -*-

import os
from parser.utils import Embedding
from parser.utils.alg import cky, crf
from parser.utils.common import bos, eos, pad, unk
from parser.utils.corpus import Corpus, Treebank
from parser.utils.field import (BertField, ChartField, Field, NGramField,
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

            self.CHART = ChartField('charts', unk=unk)
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
            elif args.feat == 'bigram':
                self.BIGRAM = NGramField('bichar', n=2, pad=pad, unk=unk,
                                         bos=bos, eos=eos, lower=True)
                self.fields = Treebank(TREE=self.TREE,
                                       CHAR=(self.CHAR, self.BIGRAM),
                                       POS=self.POS,
                                       CHART=self.CHART)
            elif args.feat == 'trigram':
                self.BIGRAM = NGramField('bichar', n=2, pad=pad, unk=unk,
                                         bos=bos, eos=eos, lower=True)
                self.TRIGRAM = NGramField('trichar', n=3, pad=pad, unk=unk,
                                          bos=bos, eos=eos, lower=True)
                self.fields = Treebank(TREE=self.TREE,
                                       CHAR=(self.CHAR, self.BIGRAM,
                                             self.TRIGRAM),
                                       POS=self.POS,
                                       CHART=self.CHART)
            else:
                self.fields = Treebank(TREE=self.TREE,
                                       CHAR=self.CHAR,
                                       POS=self.POS,
                                       CHART=self.CHART)

            train = Corpus.load(args.ftrain, self.fields)
            embed = Embedding.load(
                'data/tencent.char.200.txt',
                args.unk) if args.embed else None
            self.CHAR.build(train, args.min_freq, embed)
            if hasattr(self, 'FEAT'):
                self.FEAT.build(train)
            if hasattr(self, 'BIGRAM'):
                embed = Embedding.load(
                    'data/tencent.bi.200.txt',
                    args.unk) if args.embed else None
                self.BIGRAM.build(train, args.min_freq,
                                  embed=embed,
                                  dict_file=args.dict_file)
            if hasattr(self, 'TRIGRAM'):
                embed = Embedding.load(
                    'data/tencent.tri.200.txt',
                    args.unk) if args.embed else None
                self.TRIGRAM.build(train, args.min_freq,
                                   embed=embed,
                                   dict_file=args.dict_file)
            self.CHART.build(train, args.label_min_freq)
            self.POS.build(train)
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            self.TREE = self.fields.TREE
            if args.feat == 'bert':
                self.CHAR, self.FEAT = self.fields.CHAR
            elif args.feat == 'bigram':
                self.CHAR, self.BIGRAM = self.fields.CHAR
            elif args.feat == 'trigram':
                self.CHAR, self.BIGRAM, self.TRIGRAM = self.fields.CHAR
            else:
                self.CHAR = self.fields.CHAR
            self.POS = self.fields.POS
            self.CHART = self.fields.CHART
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        args.update({
            'n_chars': self.CHAR.vocab.n_init,
            'n_labels': len(self.CHART.vocab),
            'n_pos_labels': len(self.POS.vocab),
            'pad_index': self.CHAR.pad_index,
            'unk_index': self.CHAR.unk_index,
            'bos_index': self.CHAR.bos_index,
            'eos_index': self.CHAR.eos_index
        })

        vocab = f"{self.TREE}\n{self.CHAR}\n{self.POS}\n{self.CHART}\n"
        if hasattr(self, 'FEAT'):
            args.update({
                'n_feats': self.FEAT.vocab.n_init,
            })
            vocab += f"{self.FEAT}\n"
        if hasattr(self, 'BIGRAM'):
            args.update({
                'n_bigrams': self.BIGRAM.vocab.n_init,
            })
            vocab += f"{self.BIGRAM}\n"
        if hasattr(self, 'TRIGRAM'):
            args.update({
                'n_trigrams': self.TRIGRAM.vocab.n_init,
            })
            vocab += f"{self.TRIGRAM}\n"

        print(f"Override the default configs\n{args}")
        print(vocab[:-1])

    def train(self, loader):
        self.dp_model.train()

        for i, data in enumerate(loader):
            if self.args.feat == 'bert':
                trees, chars, feats, pos, (spans, labels) = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                trees, chars, bigram, pos, (spans, labels) = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                trees, chars, bigram, trigram, pos, (spans, labels) = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                trees, chars, pos, (spans, labels) = data
                feed_dict = {"chars": chars}

            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            span_mask = spans.gt(0)
            spans = torch.nn.functional.one_hot(spans, 5).bool()[..., 1:]
            # print(f"{labels.shape=}")
            labels_sparse = labels.to_sparse(3)
            feed_dict.update({"target": spans, "target_sparse": labels_sparse, "mask": mask})
            _, s_label, span_loss = self.dp_model(feed_dict)
            # span_mask = span_mask & mask
            # label_loss = self.criterion(s_label[span_mask], labels[span_mask])
            label_loss = self.criterion(s_label, labels_sparse.values())
            loss = (span_loss.mean() + label_loss) / self.args.update_steps
            loss.backward()
            if (i + 1) % self.args.update_steps == 0 or i == len(loader):
                nn.utils.clip_grad_norm_(self.dp_model.parameters(),
                                         self.args.clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, loader):
        self.dp_model.eval()

        total_loss = 0
        metric = BracketMetric()

        for data in loader:
            if self.args.feat == 'bert':
                trees, chars, feats, pos, (spans, labels) = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                trees, chars, bigram, pos, (spans, labels) = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                trees, chars, bigram, trigram, pos, (spans, labels) = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                trees, chars, pos, (spans, labels) = data
                feed_dict = {"chars": chars}

            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            span_mask = spans.gt(0)
            spans = torch.nn.functional.one_hot(spans, 5).bool()[..., 1:]
            feed_dict.update({"target": spans, "mask": mask})
            probs, s_label, span_loss = self.dp_model(feed_dict)
            span_mask = span_mask & mask
            loss = span_loss.mean() + \
                self.criterion(s_label[span_mask], labels[span_mask])
            preds = self.model.decode(probs, s_label, mask, self.args.marg)
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
        coarse_mask = torch.nn.functional.one_hot(torch.tensor([self.CHART.label_cluster(label) - 1 for label in self.CHART.vocab.itos]), 4).float().log()

        all_trees = []
        for data in loader:
            if self.args.feat == 'bert':
                trees, chars, feats, pos = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                trees, chars, bigram, pos = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                trees, chars, bigram, trigram, pos = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                trees, chars, pos = data
                feed_dict = {"chars": chars}
            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            feed_dict.update({"mask": mask})
            probs, s_label, _ = self.dp_model(feed_dict)
            # preds = self.model.decode(probs, s_label, mask, self.args.marg, coarse_mask.to(device=chars.device))
            preds = self.model.decode(probs, s_label, mask, self.args.marg)
            preds = [build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            all_trees.extend(preds)

        return all_trees
