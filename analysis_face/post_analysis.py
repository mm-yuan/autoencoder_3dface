import os
import sys
sys.path.append("/usr/local/micapollo01/MIC/DATA/.../")
os.environ["OMP_DYNAMIC"]="FALSE"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import scipy.io as sio
import numpy as np
import torch
import copy
from tqdm import tqdm

from scipy.io import savemat
from dataset import FaceDataset
from models import  AE
from utils import utils, dataloader
from arguments import load_args
from functions import eval_error

name = 'AE'
checkpoint_epoch = 200
latent_dims = [30, 70, 100,]
seeds = range(1, 11, 1)
studypath = '/usr/local/micapollo01/MIC/DATA/STAFF/.../'

compute_analysis = True
compute_specificity = False
compute_specificity_shapes = False
compute_orthogonality = False
compute_dimensions = False
compute_maxvar = False

device_idx = 0
torch.cuda.get_device_name(device_idx)
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

for ld in latent_dims:
    for seed in seeds:
        run = name + '_latent' + str(ld) + '/' + name + '_latent' + str(ld) + '_seed' + str(seed)

        print('Analysis for ' + run)

        path = studypath + 'runs/AE/' + run
        args = load_args(path + "/args.json")
        args.checkpoint_epoch = checkpoint_epoch
        args.nVal = None
        args.test = True
        args.eval = True
        writer = utils.Writer(args)

        # load dataset
        print("Loading data ...")
        dataloaders = dataloader.DataLoaders(args, FaceDataset(args))
        avg_shape = torch.from_numpy(dataloaders.train_loader.dataset.dataset.mean).to(device)

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
        spirals = sio.loadmat(spiral_path)['spirals'][0]
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
        if name.startswith("AE"):
            model = AE(args.in_channels, args.out_channels, args.latent_channels,
                       spirals, down_transform_list, up_transform_list).to(device)
            model = writer.load_checkpoint(model, None, None, device)
        elif name.startswith("SAE"):
            model = SAE(args.in_channels, args.out_channels, args.latent_channels,
                        spirals, down_transform_list, up_transform_list).to(device)
            model = writer.load_checkpoint(model, None, None, device)
        elif name.startswith("TBAE"):
            basemodel = AE(args.in_channels, args.out_channels, args.latent_channels,
                           spirals, down_transform_list, up_transform_list).to(device)
            tnet = TripletNet(basemodel, TripletSelector.get_selector(args), args)
            model = writer.load_checkpoint(tnet, None, None, device)
        elif name.startswith("TBSAE"):
            basemodel = SAE(args.in_channels, args.out_channels, args.latent_channels,
                            spirals, down_transform_list, up_transform_list).to(device)
            tnet = TripletNet(basemodel, TripletSelector.get_selector(args), args)
            model = writer.load_checkpoint(tnet, None, None, device)

        loss_fn = torch.nn.functional.l1_loss

        with torch.no_grad():
            if name.startswith("TB"):
                for b, [data, grp] in enumerate(dataloaders.train_loader):
                    x = data.to(device)
                    grp = grp.long().to(device)
                    _, _, emb, _ = model(x, grp)
                    if b == 0:
                        embedding = copy.deepcopy(emb)
                    else:
                        embedding = torch.cat([embedding, emb], 0)
            else:
                for b, data in enumerate(dataloaders.train_loader):
                    x = data.to(device)
                    emb, _ = model(x)
                    if b == 0:
                        embedding = copy.deepcopy(emb)
                    else:
                        embedding = torch.cat([embedding, emb], 0)
            std = torch.std(embedding, 0)
            mean = torch.mean(embedding, 0)
            del x


        def specificity(model, loader, space, device, N):
            with torch.no_grad():
                spec = np.empty(N);
                dist_to_mean = np.empty(N);
                data = torch.from_numpy(np.squeeze(loader.dataset.dataset.shapes[loader.dataset.indices, :, :])).to(
                    device)
                mean = np.mean(space.detach().cpu().numpy(), 0)
                cov = np.cov(np.transpose(space.detach().cpu().numpy()))
                for i in tqdm(range(N)):
                    sample_coeff = torch.from_numpy(np.float32(np.random.multivariate_normal(mean, cov))).to(device)
                    sample = model.decoder(sample_coeff)
                    diff = torch.min(torch.mean(torch.mean(torch.abs(data - sample), 2), 1))
                    spec[i] = diff.cpu().numpy()
                    diff2 = torch.mean(torch.mean(torch.abs(sample), 2), 1)  # already normalized no need -avgshape
                    dist_to_mean[i] = diff2.cpu().numpy()
            return spec, dist_to_mean


        def specificity_shapes(model, loader, mean, std, N):
            with torch.no_grad():
                for i in tqdm(range(N)):
                    sample_coeff = torch.normal(mean, std)
                    sample = model.decoder(sample_coeff)
                    if i == 0:
                        shapes = copy.deepcopy(sample)
                    else:
                        shapes = torch.cat([shapes, sample], 0)
                shapes = shapes.cpu().numpy()
                if hasattr(loader.dataset.dataset, 'mean'):
                    shapes += loader.dataset.dataset.mean
            return shapes


        def orthogonality(embedding):
            cov_emb = np.cov(np.transpose(embedding.detach().cpu().numpy()))
            return cov_emb


        def dimensions(model, mean, std, steps, avg_shape, device):
            with torch.no_grad():
                min = mean - 3 * std
                max = mean + 3 * std
                step = (max - min) / steps
                for s in range(steps + 1):
                    x = torch.eye(args.latent_channels[-1], device=device) * (min + s * step)
                    sp = model.decoder(x) + avg_shape
                    shape = torch.unsqueeze(sp, 0)
                    if s == 0:
                        dims = copy.deepcopy(shape)
                    else:
                        dims = torch.cat([dims, shape], 0)

                dims = dims.cpu().numpy()
            return dims


        def eval_maxvar(model, loader, loss_fn, embedding, device):
            model.eval()
            n = loader.dataset.indices.size
            ind = np.argsort(np.diag(np.cov(np.transpose(embedding.detach().cpu().numpy()))))

            with torch.no_grad():
                for b, (data, _) in enumerate(loader):
                    x = data.to(device)
                    tmp = model.encoder(x)
                    tmp[:, ind[:-3]] = 0
                    prediction = model.decoder(tmp)

                    if b == 0:
                        original = copy.deepcopy(x)
                        predictions = copy.deepcopy(prediction)
                    else:
                        original = torch.cat([original, x], 0)
                        predictions = torch.cat([predictions, prediction], 0)

                for i in range(n):
                    loss = loss_fn(predictions[i, :, :], original[i, :, :])
                    if i == 0:
                        error = copy.deepcopy(torch.unsqueeze(loss, dim=0))
                    else:
                        error = torch.cat([error, torch.unsqueeze(loss, dim=0)])

                mean_error = error.view((-1,)).mean()
                std_error = error.view((-1,)).std()
                median_error = error.view((-1,)).median()

            message = 'Error: {:.3f}+-{:.3f} | {:.3f}'.format(mean_error, std_error, median_error)
            print(message)

            original = original.cpu().numpy()
            predictions = predictions.cpu().numpy()
            if hasattr(loader.dataset.dataset, 'mean'):
                original += loader.dataset.dataset.mean
                predictions += loader.dataset.dataset.mean
                # original *= loader.dataset.dataset.std
                # predictions *= loader.dataset.dataset.std
            return original, predictions


        if compute_analysis:
            print("compute_analysis ...")
            if name.startswith("TB"):
                eval_error_tripletloss(model, dataloaders.train_loader, device, args.results_dir, 'train_',
                                       checkpoint_epoch)
                eval_error_tripletloss(model, dataloaders.test_loader, device, args.results_dir, 'test_',
                                       checkpoint_epoch)
                eval_error_tripletloss(model, dataloaders.eval_loader, device, args.results_dir, 'usuk_',
                                       checkpoint_epoch)
            else:
                eval_error(model, dataloaders.eval_loader, device, args.results_dir, 'traintest_', checkpoint_epoch)
            #    eval_error(model, dataloaders.train_loader, device, args.results_dir, 'train_', checkpoint_epoch)
            #    eval_error(model, dataloaders.test_loader, device, args.results_dir, 'test_', checkpoint_epoch)

        if compute_specificity:
            print("compute_specificity ...")
            spec, dist_to_mean = specificity(model, dataloaders.train_loader, embedding, device, 10000)

            specificity = {}
            specificity[u'specificity'] = spec
            specificity[u'dist_to_mean'] = dist_to_mean
            savemat(args.results_dir + '/specificity.mat', mdict={'specificity': specificity})

        if compute_specificity_shapes:
            print("compute_specificity_shapes ...")
            spec_shapes = specificity_shapes(model, dataloaders.train_loader, mean, std, 100)

            shapes = {}
            shapes[u'shapes'] = spec_shapes
            savemat(args.results_dir + '/specificity_shapes.mat', mdict={'shapes': shapes})

        if compute_orthogonality:
            print("compute_orthogonality ...")
            cov_emb = orthogonality(embedding)

            orthogonality = {}
            orthogonality['cov_emb'] = cov_emb
            savemat(args.results_dir + '/orthogonality.mat', mdict={'orthogonality': orthogonality})

        if compute_dimensions:
            print("compute_dimensions ...")
            dims = dimensions(model, mean, std, 5, avg_shape, device)

            dimensions = {}
            dimensions[u'dims'] = dims
            savemat(args.results_dir + '/dimensions.mat', mdict={'dimensions': dimensions})

        if compute_maxvar:
            print("compute_maxvar ...")
            orig_test, pred_test = eval_maxvar(model, dataloaders.test_loader, loss_fn, embedding, device)

            maxvar = {}
            maxvar[u'orig_test'] = orig_test
            maxvar[u'pred_test'] = pred_test
            savemat(args.results_dir + '/maxvar.mat', mdict={'maxvar': maxvar})



