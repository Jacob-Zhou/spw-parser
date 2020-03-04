# -*- coding: utf-8 -*-

from parser.utils.fn import stripe, multi_dim_max

import torch
import torch.autograd as autograd


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # count the frequency of each datapoint
    d, indices, f = x.unique(return_inverse=True, return_counts=True)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # initialize k centroids randomly
    c, old = d[torch.randperm(len(d))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # make sure number of datapoints is greater than that of clusters
    assert len(d) >= k, f"unable to assign {len(d)} datapoints to {k} clusters"

    while old is None or not c.equal(old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        mask = y.eq(torch.arange(k).unsqueeze(-1))
        # update the centroids
        c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, assigned = y[indices], y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


@torch.enable_grad()
def crf(scores, transitions, start_transitions, mask, target=None, marg=False):
    lens = mask[:, 0].sum(-1)
    total = lens.sum()
    batch_size, seq_len, seq_len, n_labels = scores.shape
    # always enable the gradient computation of scores
    # in order for the computation of marginal probs
    s = inside(scores.requires_grad_(), transitions,
               start_transitions, mask)
    lens = lens.view(1, 1, batch_size).expand([-1, n_labels, -1])
    logZ = s[0].gather(0, lens).logsumexp(1).sum()
    # marginal probs are used for decoding, and can be computed by
    # combining the inside algorithm and autograd mechanism
    # instead of the entire inside-outside process
    probs = scores
    if marg:
        probs, = autograd.grad(
            logZ, scores, retain_graph=scores.requires_grad)
    if target is None:
        return None, probs
    s = inside(scores.requires_grad_(), transitions,
               start_transitions, mask, target)
    score = s[0].gather(0, lens).logsumexp(1).sum()
    loss = (logZ - score) / total
    return loss, probs


def inside(scores, transitions, start_transitions, mask, cands=None):
    batch_size, seq_len, seq_len, n_labels = scores.shape
    # [seq_len, seq_len, n_labels, batch_size]
    scores = scores.permute(1, 2, 3, 0)
    # [seq_len, seq_len, n_labels, batch_size]
    mask = mask.permute(1, 2, 0)
    if cands is not None:
        cands = cands.permute(1, 2, 3, 0)
        cands = cands & mask.view(seq_len, seq_len, 1, batch_size)
        scores = scores.masked_fill(~cands, -1e36)
    s = torch.full_like(scores, float('-inf'))

    start_transitions = start_transitions.view(1, 1, n_labels)
    transitions = transitions.view(1, 1,
                                   n_labels, n_labels, n_labels)

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        # diag_mask is used for ignoring the excess of each sentence
        diag_mask = mask.diagonal(w)
        # [batch_size, n, n_labels]
        emit_scores = scores.diagonal(w).permute(1, 2, 0)[diag_mask]
        diag_s = s.diagonal(w).permute(1, 2, 0)
        if w == 1:
            diag_s[diag_mask] = emit_scores + start_transitions
            continue
        # [batch_size, n, w-1, n_labels, n_labels, n_labels]
        emit_scores = emit_scores.view(-1, 1, 1, 1, n_labels)
        s_left = stripe(s, n, w-1, (0, 1)).permute(3, 0, 1, 2).contiguous()
        s_left = s_left.view(batch_size,
                             n, w-1,
                             n_labels, 1, 1)[diag_mask]
        s_right = stripe(s, n, w-1, (1, w), 0).permute(3, 0, 1, 2).contiguous()
        s_right = s_right.view(batch_size,
                               n, w-1,
                               1, n_labels, 1)[diag_mask]
        # [*, w-1, n_labels, n_labels, n_labels]
        inner = s_left + s_right + transitions + emit_scores
        # [*, n_labels]
        inner = inner.logsumexp([1, 2, 3])
        diag_s[diag_mask] = inner
    return s


def cky(scores, transitions, start_transitions, mask):
    lens = mask[:, 0].sum(-1)
    batch_size, seq_len, seq_len, n_labels = scores.shape
    # [seq_len, seq_len, n_labels, batch_size]
    scores = scores.permute(1, 2, 3, 0)
    s = torch.zeros_like(scores)
    bp = scores.new_zeros(seq_len, seq_len, n_labels, batch_size, 3).long()

    start_transitions = start_transitions.view(n_labels, 1, 1)
    transitions = transitions.view(1, 1,
                                   n_labels, n_labels, n_labels,
                                   1)

    for w in range(1, seq_len):
        n = seq_len - w
        starts = bp.new_tensor(range(n)).unsqueeze(0)
        # [n_labels, batch_size, n]
        emit_scores = scores.diagonal(w)

        if w == 1:
            # [n_labels, batch_size, n]
            s.diagonal(w).copy_(emit_scores + start_transitions)
            continue
        # [n, w-1, n_labels, n_labels, n_labels, batch_size]
        emit_scores = emit_scores.permute(2, 0, 1).contiguous().view(n, 1,
                                                                     1, 1, n_labels,
                                                                     batch_size)
        s_left = stripe(s, n, w-1, (0, 1)).view(n, w-1,
                                                n_labels, 1, 1,
                                                batch_size)
        s_right = stripe(s, n, w-1, (1, w), 0).view(n, w-1,
                                                    1, n_labels, 1,
                                                    batch_size)
        inner = s_left + s_right + transitions + emit_scores
        # [n_labels, batch_size, n, w-1, n_labels, n_labels]
        inner = inner.permute(4, 5, 0, 1, 2, 3)
        # [n_labels, batch_size, n]
        inner, idx = multi_dim_max(inner, [3, 4, 5])
        idx[..., 0] = idx[..., 0] + starts + 1
        # [n_labels, batch_size, n, 3]
        idx = idx.permute(0, 1, 3, 2)
        s.diagonal(w).copy_(inner)
        # [n_labels, batch_size, 3, n]
        bp.diagonal(w).copy_(idx)

    def backtrack(bp, label, i, j):
        if j == i + 1:
            return [(i, j, label)]
        split, llabel, rlabel = bp[i][j][label]
        ltree = backtrack(bp, llabel, i, split)
        rtree = backtrack(bp, rlabel, split, j)
        return [(i, j, label)] + ltree + rtree

    labels = s.permute(3, 0, 1, 2).argmax(-1)
    bp = bp.permute(3, 0, 1, 2, 4).tolist()
    trees = [backtrack(bp[i], labels[i, 0, length], 0, length)
             for i, length in enumerate(lens.tolist())]

    return trees


def simple_cky(scores, mask):
    lens = mask[:, 0].sum(-1)
    scores = scores.sum(-1)
    scores = scores.permute(1, 2, 0)
    seq_len, seq_len, batch_size = scores.shape
    s = scores.new_zeros(seq_len, seq_len, batch_size)
    p = scores.new_zeros(seq_len, seq_len, batch_size).long()

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p.new_tensor(range(n)).unsqueeze(0)

        if w == 1:
            s.diagonal(w).copy_(scores.diagonal(w))
            continue
        # [n, w, batch_size]
        s_span = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
        # [batch_size, n, w]
        s_span = s_span.permute(2, 0, 1)
        # [batch_size, n]
        s_span, p_span = s_span.max(-1)
        s.diagonal(w).copy_(s_span + scores.diagonal(w))
        p.diagonal(w).copy_(p_span + starts + 1)

    def backtrack(p, i, j):
        if j == i + 1:
            return [(i, j)]
        split = p[i][j]
        ltree = backtrack(p, i, split)
        rtree = backtrack(p, split, j)
        return [(i, j)] + ltree + rtree

    p = p.permute(2, 0, 1).tolist()
    trees = [backtrack(p[i], 0, length)
             for i, length in enumerate(lens.tolist())]

    return trees


def heatmap(corr, name='matrix'):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(40, 40))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    cmap = "RdBu"
    for i in range(4):
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr[..., i], cmap=cmap, center=-1, ax=ax[i//2][i % 2],
                    square=True, linewidths=.5,
                    xticklabels=False, yticklabels=False,
                    cbar=False, vmax=1.5, vmin=-1.5)
    plt.savefig(f'{name}.png')
    plt.close()

    # heatmap(s_span[0].cpu().detach(),
    #     torch.ones(seq_len-1,seq_len-1).triu_(1).eq(0))
