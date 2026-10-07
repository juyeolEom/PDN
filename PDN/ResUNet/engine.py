from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import create_run_directory, load_weights, save_weights
from .config import save_config
from .dataloader import trainset, testset
from .pulse_denoise_net import PDN
from .sampling import generate_mask_pair, generate_subimages
from .utils import build_generator, build_mask_generator, get_device, get_device_ids, seed_everything, seed_worker, seed_noise_worker


def build_model(opt, device):
    net = PDN(in_ch=1, filters=[16, 32, 64, 128])
    net.to(device)

    device_ids = get_device_ids(opt.GPU)
    if len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids)

    return net


def build_trainloader(opt):
    train_data = trainset(opt)
    worker_init = (seed_noise_worker if train_data.additive_noise.enabled
                   else partial(seed_worker, base_seed=opt.seed))
    return DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        worker_init_fn=worker_init,
        generator=build_generator(opt.seed),
    )


def train_epoch(net, trainloader, optimizer, scheduler, l1, device, mask_generator=None, st_mixing=False):
    net.train()
    train_loss = 0.0

    for noisy in trainloader:
        f_pulse = noisy.to(device)  # b x c(=1) x pulse_length x patch_x x patch_y
        mask1, mask2, mask3 = generate_mask_pair(f_pulse, generator=mask_generator, st_mixing=st_mixing)
        sub_1 = generate_subimages(f_pulse, mask1)
        sub_2 = generate_subimages(f_pulse, mask2)
        sub_3 = generate_subimages(f_pulse, mask3)

        out = net(sub_1)

        loss2neighbor_1 = 0.5 * l1(sub_2, out) + 0.5 * l1(sub_3, out)

        total_loss = loss2neighbor_1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    scheduler.step(train_loss)

    return train_loss


def run_train(opt):
    seed_everything(opt.seed)

    device = get_device(opt.GPU)
    net = build_model(opt, device)
    l1 = nn.L1Loss().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=opt.lr_decay,
        patience=opt.lr_decay_patience,
    )

    trainloader = build_trainloader(opt)

    mask_seed = opt.seed if opt.mask_seed is None else opt.mask_seed
    mask_generator = build_mask_generator(mask_seed, device)

    if opt.phase == 'augment':
        load_weights(opt.weight_load_path, net, device)
    epochs = 500 if opt.phase == 'augment' else opt.n_epochs
    for epoch in range(epochs):
        train_loss = train_epoch(
            net, trainloader, optimizer, scheduler, l1, device,
            mask_generator=mask_generator, st_mixing=opt.st_mixing,
        )
        print(f'[Epoch {epoch + 1:04d}/{epochs:04d}] [Loss: {train_loss:.6f}]', flush=True)
    output_path = create_run_directory(opt.weight_save_path) / 'weights.pth'
    save_weights(net, output_path)
    save_config(opt, output_path.parent)
    print(f'[Saved] {output_path}')


def test_epoch(net, opt, device):
    load_weights(opt.weight_load_path, net, device)

    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    roi_size = opt.patch_x // 2
    grid = opt.test_grid
    test_data = testset(opt, grid)
    pulse_out = np.zeros(test_data.pulse_data.shape, dtype=np.float32)
    testloader = DataLoader(test_data, batch_size=1, shuffle=False)

    with torch.no_grad():
        for iteration, noisy in enumerate(testloader):
            noisy_input, mean_ = noisy
            noisy_input = noisy_input.to(device)
            noisy_output = net(noisy_input).detach().cpu().numpy()

            x_roi = iteration // (grid * 2 - 1)
            y_roi = iteration % (grid * 2 - 1)
            tmp = noisy_output[0, 0] + mean_.item()

            last_roi = grid * 2 - 2
            if x_roi > 0:
                for k in range(roi_size // 2):
                    tmp[:, k, :] *= (2 * k / roi_size)
            if x_roi < last_roi:
                for k in range(roi_size // 2, roi_size):
                    tmp[:, k, :] *= (2 - 2 * k / roi_size)
            if y_roi > 0:
                for k in range(roi_size // 2):
                    tmp[:, :, k] *= (2 * k / roi_size)
            if y_roi < last_roi:
                for k in range(roi_size // 2, roi_size):
                    tmp[:, :, k] *= (2 - 2 * k / roi_size)

            x0 = x_roi * (roi_size // 2)
            y0 = y_roi * (roi_size // 2)
            pulse_out[:, x0:x0 + roi_size, y0:y0 + roi_size] += tmp

    output_path = create_run_directory(Path(opt.weight_load_path).parent) / (Path(opt.data_path).stem + '_denoised.npy')
    np.save(output_path, pulse_out)
    save_config(opt, output_path.parent)
    print(f'[Saved] {output_path}')


def run_test(opt):
    seed_everything(opt.seed)
    device = get_device(opt.GPU)
    net = build_model(opt, device)
    test_epoch(net, opt, device)
