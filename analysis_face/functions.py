import time
import os
import torch
from tqdm import tqdm
import copy
import scipy.io as sio
from sklearn.manifold import TSNE


def run(model, train_loader, val_loader, loss_fn, epochs, optimizer, scheduler,
        writer, viz, device, start_epoch=1):

    if 'SAE' in type(model).__name__:
        print('initializing SVD...')
        if model.emb is None:
            with torch.no_grad():
                for b, data in enumerate(train_loader):
                    x = data.to(device)
                    emb = model.encode_layers(x)

                    if b == 0:
                        embedding = copy.deepcopy(emb)
                    else:
                        embedding = torch.cat([embedding, emb], 0)
                model.initialize_svd(embedding)
            del x, embedding

    for epoch in range(start_epoch, epochs + 1):
        t = time.time()
        train_loss = train(model, optimizer, train_loader, loss_fn, device)
        t_duration = time.time() - t
        val_loss = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'validation_loss': val_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch, 0)
        if epoch % 50 == 0:
            writer.save_checkpoint(model, optimizer, scheduler, epoch, epoch)
        viz.line(torch.ones((1,)).cpu() * epoch, torch.Tensor([train_loss, val_loss]).unsqueeze(0).cpu(),
                 'Loss', ['Training loss', 'Validation loss'])
        viz.save()


def train(model, optimizer, loader, loss_fn, device):
    model.train()

    total_loss = 0
    for data in tqdm(loader):
        optimizer.zero_grad()
        x = data.to(device)
        z, out = model(x)
        loss = loss_fn(out, x)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return (total_loss / len(loader))


def validate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for data in loader:
            x = data.to(device)
            _, pred = model(x)
            total_loss += loss_fn(pred, x)
    return (total_loss / len(loader))


def eval_error(model, loader, device, results_dir, fname, epoch, viz=None, plot_emb=False):
    model.eval()

    with torch.no_grad():
        for b, data in enumerate(loader):
            x = data.to(device)
            emb, prediction = model(x)

            if b == 0:
                original = copy.deepcopy(x)
                predictions = copy.deepcopy(prediction)
                embedding = copy.deepcopy(emb)
            else:
                original = torch.cat([original, x], 0)
                predictions = torch.cat([predictions, prediction], 0)
                embedding = torch.cat([embedding, emb], 0)

    original = original.cpu().numpy()
    predictions = predictions.cpu().numpy()
    embedding = embedding.cpu().numpy()
    sio.savemat(os.path.join(results_dir, fname+'predictions_epoch{0}.mat'.format(epoch)), {'original': original,
                                                               'predicted': predictions,'embedding': embedding})

    if plot_emb:
        tsne_emb = TSNE(n_components=3).fit_transform(embedding)
        viz.scatter(tsne_emb, 'Embedding')
        viz.save()



