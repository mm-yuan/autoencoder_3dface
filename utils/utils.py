import torch
import os
import numpy as np
from glob import glob
from visdom import Visdom
import json
import time

### FUNCTIONS

def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))

### CLASSES

class VisdomPlotter(object):
    """Plots to Visdom"""
    def __init__(self, log_dir, env_name='main', port=8668, enabled=True):
        self.log_dir = log_dir
        self.env_name = env_name
        self.enabled = enabled
        if enabled:
            try:
                self.viz = Visdom(
                    log_to_filename=log_dir+'/visdom', env=env_name, port=port, raise_exceptions=True)
            except (ConnectionError, ConnectionRefusedError):
                print('Visdom server not found: logging offline')
                self.viz = Visdom(
                    log_to_filename=log_dir + '/visdom', env=env_name, offline=True)  # use vis.replay_log to load
                self.viz.use_socket = False
        else:
            self.viz = None
        self.data = dict()
        self.plots = {}

    def save(self):
        self.viz.save([self.viz.env])

    def line(self, x, y, var_name, legend, dash=None, color=None):
        if self.enabled:
            if var_name not in self.plots:
                self.plots[var_name] = self.viz.line(X=x, Y=y, opts=dict(
                    legend=legend,
                    title=var_name,
                    xlabel='Epochs',
                    ylabel=var_name,
                    dash=dash,
                    linecolor=color
                ))
            else:
                self.viz.line(X=x, Y=y, win=self.plots[var_name],
                              update='append')

    def scatter(self, x, var_name, m_size=3, mb_size=0.1, labels=None):
        self.plots[var_name] = self.viz.scatter(X=x, opts=dict(
            title=var_name,
            markersize=m_size,
            markerborderwidth=mb_size,
            textlabels=labels
        ))

    def matplot(self,plt):
        self.viz.matplot(plt)

    def text(self,txt, win=None, append=True):
        if(win==None):
            return self.viz.text(txt)
        else:
            return self.viz.text(txt, win, append)


class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, Validation Loss: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_loss'], info['validation_loss'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_model(self, model, params=None, optimizer=None):
        with open(self.log_file, 'a') as log_file:
            log_file.write('model: ' + str(model) + '\n\n')
            if params: log_file.write('# params: ' + str(params) + '\n\n')
            if optimizer: log_file.write('optimizer: ' + str(optimizer) + '\n\n')

    def save_checkpoint(self, model, optimizer, scheduler, epoch, n):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(n)))

    def load_checkpoint(self, model, device, optimizer=None, scheduler=None, resume=False, path=None):
        if not path:
            path = os.path.join(self.args.checkpoints_dir,
                             'checkpoint_{:03d}.pt'.format(self.args.checkpoint_epoch))
        print('loading checkpoint from file %s' % path)
        checkpoint_dict = torch.load(path,  map_location=device)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        if resume:
            epoch = checkpoint_dict['epoch'] + 1
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            return epoch, model, optimizer, scheduler
        else:
            return model
