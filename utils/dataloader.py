from torch.utils.data import Dataset, DataLoader, Subset
import scipy.io as sio
import numpy as np
import torch
import random

class DataLoaders:
    """ Creates mutual exclusive loaders for training, evaluation and testing backed by the same dataset. """

    def __init__(self, args, dataset):
    
        tmp = sio.loadmat(args.sets_ind)
        kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

        g = torch.Generator()
        g.manual_seed(args.seed)

        if (args.train == True):
            if args.nVal:
                train_ind = tmp['train_ind'][:-args.nVal] - 1
                train_set = Subset(dataset, train_ind)

                val_ind = tmp['train_ind'][-args.nVal:] - 1
                val_set = Subset(dataset, val_ind)

                # DataLoader used for training in batches:
                self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
                # DataLoader used for evaluating:
                self.val_loader = DataLoader(val_set, batch_size=args.batch_size, **kwargs)
            else:
                train_ind = tmp['train_ind'] - 1
                train_set = Subset(dataset, train_ind)
                # DataLoader used for training in batches:
                self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

        if (args.test == True):
            test_set = Subset(dataset, tmp['test_ind'] - 1)
            self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

        if (args.eval == True):
            if args.name.startswith("TB"):
                eval_set = Subset(dataset, tmp['usuk_ind'] - 1)
                self.eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=args.shuffle)
            else:
                self.eval_loader = DataLoader(dataset, batch_size=args.batch_size,shuffle=args.shuffle)
