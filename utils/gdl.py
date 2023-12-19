import torch
import torch.nn as nn
import torch.nn.functional as F


def Pool(x, trans, spiral=None, mode=None):
    if mode == 'max':
        out = torch.empty(x.shape[0], trans.shape[0], x.shape[-1])
        _, col = trans._indices()
        for i, c in enumerate(col):
            tmp, _ = torch.max(x[:, spiral[c], :], 1)
            out[:, i, :] = tmp

    elif mode == 'min':
        out = torch.empty(x.shape[0], trans.shape[0], x.shape[-1])
        _, col = trans._indices()
        for i, c in enumerate(col):
            tmp, _ = torch.min(x[:, spiral[c], :], 1)
            out[:, i, :] = tmp

    elif mode == 'mean':
        out = torch.empty(x.shape[0], trans.shape[0], x.shape[-1])
        _, col = trans._indices()
        for i, c in enumerate(col):
            tmp = torch.mean(x[:, spiral[c], :], 1)
            out[:, i, :] = tmp

    else:
        row, col = trans._indices()
        value = trans._values().unsqueeze(-1)
        out = torch.index_select(x, 1, col) * value
        out = scatter_add(out, row, 1, dim_size=trans.size(0))
    return out


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, pool=None, act=True):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.pool = pool
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = self.conv(x)
        if self.act:
            out = F.elu(out)
        out = Pool(out, down_transform, self.conv.indices, self.pool)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, act=True):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = self.conv(out)
        if self.act:
            out = F.elu(out)
        return out


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src
