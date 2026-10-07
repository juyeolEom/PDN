import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .additive_noise import RandomAdditiveNoise

def load_pulse(path):
    pulse = np.load(path, allow_pickle=False)
    if pulse.ndim != 3 or pulse.shape[0] != 128:
        raise ValueError('Expected data shape (128, height, width).')
    if not np.issubdtype(pulse.dtype, np.floating) or not np.isfinite(pulse).all():
        raise ValueError('Data must contain finite floating-point values.')
    return pulse.astype(np.float32, copy=False)


def random_transform(input):
    p_trans = random.randrange(8)
    if p_trans == 0:
        input = input
    elif p_trans == 1:
        input = np.rot90(input, k=1, axes=(1, 2))
    elif p_trans == 2:
        input = np.rot90(input, k=2, axes=(1, 2))
    elif p_trans == 3:
        input = np.rot90(input, k=3, axes=(1, 2))
    elif p_trans == 4:
        input = input[:, :, ::-1]
    elif p_trans == 5:
        input = input[:, :, ::-1]
        input = np.rot90(input, k=1, axes=(1, 2))
    elif p_trans == 6:
        input = input[:, :, ::-1]
        input = np.rot90(input, k=2, axes=(1, 2))
    elif p_trans == 7:
        input = input[:, :, ::-1]
        input = np.rot90(input, k=3, axes=(1, 2))
    return input


def random_roll(input):
    r_trans = np.random.randint(-2, 2)
    input = np.roll(input, r_trans, axis=0)
    return input


class trainset(Dataset):
    def __init__(self, opt):
        self.path = opt.data_path
        self.patch_x = opt.patch_x
        self.multiplication = opt.train_dataset_size

        self.pulse_data = load_pulse(self.path)

        if min(self.pulse_data.shape[1:]) < self.patch_x:
            raise ValueError('Training data must be at least patch_x in both spatial dimensions.')
        self.additive_noise = RandomAdditiveNoise(
            opt.noise_min if opt.phase == 'augment' else 0.0,
            opt.noise_max if opt.phase == 'augment' else 0.0,
        )

    def __getitem__(self, index):
        pulse_data = self.pulse_data

        x = np.random.randint(0, pulse_data.shape[1] - self.patch_x + 1, 1)[0]
        y = np.random.randint(0, pulse_data.shape[2] - self.patch_x + 1, 1)[0]
        pulse_data = pulse_data[:, x:x+self.patch_x, y:y+self.patch_x]

        pulse_data = pulse_data-pulse_data.mean()

        pulse_data = random_transform(pulse_data)
        pulse_data = random_roll(pulse_data)

        # After centering/augmentation, before engine.py creates spatial pairs.
        pulse_data = self.additive_noise(pulse_data)

        pulse_data = torch.from_numpy(np.expand_dims(pulse_data, 0).copy())
        return pulse_data

    def __len__(self):
        return self.multiplication


class testset(Dataset):
    def __init__(self, opt, grid=4):
        self.path = opt.data_path
        self.patch_x = opt.patch_x // 2
        self.grid = grid
        self.multiplication = (self.grid * 2 - 1) * (self.grid * 2 - 1)

        self.pulse_data = load_pulse(self.path)
        expected = self.grid * self.patch_x
        if self.pulse_data.shape[1:] != (expected, expected):
            raise ValueError(f'Test data spatial shape must be ({expected}, {expected}); adjust test_grid and patch_x.')


    def __getitem__(self, index):
        pulse_data = self.pulse_data

        x = index // (self.grid * 2 - 1) * self.patch_x // 2
        y = index % (self.grid * 2 - 1) * self.patch_x // 2

        pulse_data = pulse_data[:, x:x+self.patch_x, y:y+self.patch_x]
        mean_ = pulse_data.mean()
        pulse_data = pulse_data - mean_

        pulse_data = torch.from_numpy(np.expand_dims(pulse_data, 0).copy())
        return pulse_data, mean_

    def __len__(self):
        return self.multiplication
