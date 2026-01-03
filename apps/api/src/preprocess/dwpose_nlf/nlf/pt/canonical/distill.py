import itertools

import numpy as np
import simplepyutils as spu
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader, TensorDataset

from nlf.paths import PROJDIR
from nlf.pt.models.field import GPSNet


def main():
    out_dim = 1024
    hidden_dim = 2048
    pos_enc_dim = 512
    n_steps = 200000
    batch_size = 1024

    x = np.load(f'{PROJDIR}/canonical_nodes3.npy').astype(np.float32)
    y = np.load(f'{PROJDIR}/canonical_eigvec3.npy')[:, 1 : out_dim + 1].astype(np.float32)
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = itertools.chain.from_iterable(
        itertools.repeat(
            DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True
            ),
        )
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    model = GPSNet(pos_enc_dim=pos_enc_dim, hidden_dim=hidden_dim, output_dim=out_dim)
    train_model(model, train_loader, val_loader, n_steps)
    torch.save(
        model.state_dict(), f'{PROJDIR}/lbo_mlp_{pos_enc_dim}fourier_{hidden_dim}gelu_{out_dim}.pt'
    )


def train_model(model, train_loader, val_loader, n_steps, validation_freq=1000):
    device = torch.device('cuda')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=n_steps)
    loss_fn = nn.MSELoss()

    model.train()
    pbar = spu.progressbar(total=n_steps)
    postfix_dict = dict()
    for step, (x_batch, y_batch) in enumerate(itertools.islice(train_loader, n_steps)):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)

        with torch.no_grad():
            metrics = compute_metrics(pred, y_batch)
            postfix_dict.update(metrics)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % validation_freq == 0:
            val_loss = validate_model(model, val_loader, device)
            postfix_dict['val_loss'] = val_loss
            model.train()

        pbar.update()
        postfix_dict['loss'] = loss.item()
        pbar.set_postfix({k: f'{v:.3f}' for k, v in postfix_dict.items()})


def validate_model(model, val_loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            total_loss += loss_fn(pred, y_batch).item()

    return total_loss / len(val_loader)


def compute_metrics(preds, targets):
    def l2_norm(x):
        return torch.sqrt(torch.mean(torch.square(x))).item()

    return {
        'l2': l2_norm(preds - targets),
        'l2_10': l2_norm(preds[:, :10] - targets[:, :10]),
        'l2_100': l2_norm(preds[:, :100] - targets[:, :100]),
        'l2_over128': l2_norm(preds[:, 128:256] - targets[:, 128:256]),
        'l2_over256': l2_norm(preds[:, 256:512] - targets[:, 256:512]),
        'l2_over512': l2_norm(preds[:, 512:] - targets[:, 512:]),
    }


if __name__ == '__main__':
    main()
