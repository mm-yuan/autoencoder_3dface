# Utility functions & configuration constants
import os
import argparse
import json
import torch.cuda

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Autoencoder')
    
    # general
    parser.add_argument('--name', default='AE', type=str, help='name of experiment')
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--seed', default=6, type=int, help='set random seed')
    parser.add_argument('--train', default=True, type=bool, help='training mode')
    parser.add_argument('--test', default=False, type=bool, help='test mode')
    parser.add_argument('--eval', default=False, type=bool, help='test mode')
    parser.add_argument('--resume', default=False, type=bool, help='resuming from checkpoint')
    parser.add_argument('--resume2', default=False, type=bool, help='resuming from checkpoint')
    parser.add_argument('--checkpoint_epoch', default=0, type=int)
    parser.add_argument('--cuda', action='store_true', default=True and torch.cuda.is_available(),
                        help='enables CUDA training')
    base = '/usr/local/micapollo01/MIC/DATA/STAFF/myuan0/projects/PhenotypingMethods/'
    # data
    parser.add_argument('--data_dir',
                        default=base + '/GDL/data/face/USUK_FACE.mat',
                        type=str, metavar='PATH', help='data path')
    parser.add_argument('--sets_ind',
                        default=base + '/GDL/data/face/traintest9010.mat',
                        type=str, metavar='PATH', help='indices fot train, validation and test set')
    parser.add_argument('--nVal', default=200, help='# validation samples')
    parser.add_argument('--sample_dir',
                        default=base+'/GDL/data/face/meshlab_sample/',
                        type=str, metavar='PATH', help='directory containing downsample matrices')
    parser.add_argument('--normalization', default=True, type=bool, help='normalize shapes for training')
    parser.add_argument('--mean_center', default=True, type=bool, help='subtract avg from faces')

    # Visdom
    parser.add_argument('--visdom', default=True, type=bool, help='plot training curves in visdom')
    parser.add_argument('--visdom_port', default=8668, type=int)
    
    # training hyperparameters
    parser.add_argument('--batch_size', default=80, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--shuffle', default=False, type=bool)
    parser.add_argument('--weighted', default=True, type=bool)
    
    # network hyperparameters
    parser.add_argument('--out_channels', default=[64, 64, 64, 64], nargs='+', type=int)
    parser.add_argument('--latent_channels', default=[1024, 100], type=int)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--spiral_length', default=[19, 19, 6, 6], type=int, nargs='+')
    parser.add_argument('--dilation', default=[1, 1, 1, 1], type=int, nargs='+')
    parser.add_argument('--pooling', default='mean', type=str)

    # optimizer hyperparmeters
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_decay', default=0.99, type=float)
    parser.add_argument('--decay_step', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-6, type=float)

    args = parser.parse_args()

    return args

def save_args(args, path):
    """Saves parameters to json file"""
    json_path = "{}/args.json".format(path)
    with open(json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(json_path):
    """Loads parameters from json file"""
    with open(json_path) as f:
        params = json.load(f)
    return argparse.Namespace(**params)
