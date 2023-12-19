import os
import sys
sys.path.append("/usr/local/micapollo01/MIC/DATA/.../") # your path
os.environ["OMP_DYNAMIC"]="FALSE"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # the gpu you reserved
import argparse
import scipy.io as sio
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dataset import FaceDataset
from models import AE
from functions import run
from utils import utils, dataloader, generate_spiral_seq
from arguments import parse_args, save_args


latent_dims = [30, 70]


def main(latent_dim, seed):
    args = parse_args()
    # loop through a set of latent_dim, seed
    args.seed = seed
    args.latent_channels[-1] = latent_dim
    modelname = args.name + '_latent' + str(latent_dim)

    args.work_dir = os.path.dirname(os.path.realpath(__file__))
    args.out_dir = os.path.join(args.work_dir, 'runs/AE/' + modelname + '/' + modelname + '_seed' + str(args.seed))
    args.checkpoints_dir = os.path.join(args.out_dir, 'checkpoints')
    args.results_dir = os.path.join(args.out_dir, 'results')
    args.visdom_dir = os.path.join(args.work_dir, 'visdom')
    print(args)

    utils.makedirs(args.out_dir)
    utils.makedirs(args.checkpoints_dir)
    utils.makedirs(args.results_dir)
    utils.makedirs(args.visdom_dir)
    save_args(args, args.out_dir)

    writer = utils.Writer(args)
    viz = utils.VisdomPlotter(args.out_dir, modelname + '_seed' + str(args.seed), args.visdom_port, args.visdom)
    # initialize visdom before running this script
    # python -m visdom.server -env_path /usr/local/micapollo01/.../visdom -port 6666

    device_idx = 0
    torch.cuda.get_device_name(device_idx)
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

    # deterministic
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False  # will benchmark several algorithms and pick the fastest, rule of thumb: useful if you have fixed input sizes
    cudnn.deterministic = True

    # load dataset
    print("Loading data ...")
    dataloaders = dataloader.DataLoaders(args, FaceDataset(args))

    # load transform matrices
    print("Loading transformation matrices ...")
    meshdata = sio.loadmat(os.path.join(args.sample_dir + 'meshdata.mat'))
    A = np.ndarray.tolist(meshdata['A'])[0]
    D = np.ndarray.tolist(meshdata['D'])[0]
    U = np.ndarray.tolist(meshdata['U'])[0]
    F = np.ndarray.tolist(meshdata['F'])[0]

    M = sio.loadmat(os.path.join(args.sample_dir + 'M.mat'))
    M['v'] = np.ndarray.tolist(M['v'])[0]
    M['f'] = np.ndarray.tolist(M['f'])[0]

    spiral_path = args.sample_dir + 'spirals' + ''.join([str(elem) for elem in args.spiral_length]) + '.mat'
    if os.path.exists(spiral_path):
        spirals = sio.loadmat(spiral_path)['spirals'][0]
    else:
        spirals = [
            np.asarray(generate_spiral_seq.extract_spirals(M['v'][idx], A[idx],
                                                           args.spiral_length[idx], args.dilation[idx]))
            for idx in range(len(args.spiral_length))
        ]
        spirals = np.asarray(spirals)
        sio.savemat(spiral_path, {'spirals': spirals})
    spirals = spirals.tolist()
    spirals = [torch.tensor(elem).contiguous().to(device) for elem in spirals]

    down_transform_list = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in D
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in U
    ]

    model = AE(args.in_channels, args.out_channels, args.latent_channels,
               spirals, down_transform_list, up_transform_list).to(device)
    params = utils.count_parameters(model)
    print('Number of parameters: {}'.format(params))
    print(model)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                args.decay_step,
                                                gamma=args.lr_decay)
    writer.print_model(model, params, optimizer)

    loss_fn = torch.nn.functional.l1_loss

    if args.resume:
        start_epoch, model, optimizer, scheduler = writer.load_checkpoint(model, device, optimizer, scheduler, True)
        print('Resuming from epoch %s' % (str(start_epoch)))
        run(model, dataloaders.train_loader, dataloaders.val_loader, loss_fn, args.epochs, optimizer, scheduler,
            writer, viz, device, start_epoch)
    else:
        run(model, dataloaders.train_loader, dataloaders.val_loader, loss_fn, args.epochs, optimizer, scheduler,
            writer, viz, device)


if __name__ == '__main__':
    for i in latent_dims:
        tmp_ld = i
        for j in range(10, 21, 1):
            tmp_seed = j
            main(tmp_ld, tmp_seed)


