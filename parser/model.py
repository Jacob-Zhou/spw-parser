# -*- coding: utf-8 -*-

from parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM, TreeCRFLoss
from parser.modules.dropout import IndependentDropout, SharedDropout
from parser.utils.alg import simple_cky

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        # the embedding layer
        n_feat_embed = 0
        self.word_embed = nn.Embedding(num_embeddings=args.n_words,
                                       embedding_dim=args.n_embed)
        if args.feat == 'char':
            self.feat_embed = CHAR_LSTM(n_chars=args.n_feats,
                                        n_embed=args.n_char_embed,
                                        n_out=args.n_feat_embed)
            n_feat_embed = args.n_feat_embed
        elif args.feat == 'bert':
            self.feat_embed = BertEmbedding(model=args.bert_model,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.n_feat_embed)
            n_feat_embed = args.n_feat_embed

        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=args.n_embed+n_feat_embed,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        self.mlp_span_l = MLP(n_in=args.n_lstm_hidden*2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_span_r = MLP(n_in=args.n_lstm_hidden*2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_label_l = MLP(n_in=args.n_lstm_hidden*2,
                               n_out=args.n_mlp_label,
                               dropout=args.mlp_dropout)
        self.mlp_label_r = MLP(n_in=args.n_lstm_hidden*2,
                               n_out=args.n_mlp_label,
                               dropout=args.mlp_dropout)

        # the Biaffine layers
        self.span_attn = Biaffine(n_in=args.n_mlp_span,
                                  n_out=2,
                                  bias_x=True,
                                  bias_y=False)
        self.label_attn = Biaffine(n_in=args.n_mlp_label,
                                   n_out=args.n_labels,
                                   bias_x=True,
                                   bias_y=True)

        self.crf = TreeCRFLoss(2, True)
        self.cluster_bias = nn.Linear(
            in_features=2,
            out_features=args.n_labels,
            bias=False)
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def forward(self, feed_dict):
        words = feed_dict["words"]
        feats = feed_dict["feats"]
        target = feed_dict.get("target", None)
        mask = feed_dict["mask"]
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        word_mask = words.ne(self.pad_index)
        lens = word_mask.sum(dim=1)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        if self.args.feat == 'char':
            feat_embed = self.feat_embed(feats[word_mask])
            feat_embed = pad_sequence(feat_embed.split(lens.tolist()), True)
            word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
            # concatenate the word and feat representations
            embed = torch.cat((word_embed, feat_embed), dim=-1)
        elif self.args.feat == 'bert':
            feat_embed = self.feat_embed(*feats)
            word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
            # concatenate the word and feat representations
            embed = torch.cat((word_embed, feat_embed), dim=-1)
        else:
            embed = self.embed_dropout(word_embed)[0]

        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        x_f, x_b = x.chunk(2, dim=-1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        # apply MLPs to the BiLSTM output states
        span_l = self.mlp_span_l(x)
        span_r = self.mlp_span_r(x)
        label_l = self.mlp_label_l(x)
        label_r = self.mlp_label_r(x)

        # [batch_size, seq_len, seq_len, 4]
        s_span = self.span_attn(span_l, span_r).permute(0, 2, 3, 1)
        loss, s_span = self.crf(s_span, mask, target)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)
        s_label = s_label + self.cluster_bias(s_span.detach())

        return s_span, s_label, loss.view(1) if loss is not None else None

    def decode(self, s_span, s_label, mask):
        pred_spans = simple_cky(s_span, mask)
        pred_labels = s_label.argmax(-1).tolist()
        preds = [[(i, j, labels[i][j]) for i, j, _ in spans]
                 for spans, labels in zip(pred_spans, pred_labels)]

        return preds

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)


def heatmap(corr, name='matrix'):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(200, 4))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    cmap = "RdBu"

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, center=0, ax=ax,
                square=True, linewidths=.5,
                xticklabels=False, yticklabels=False,
                cbar=False)
    plt.savefig(f'{name}.png')
    plt.close()
