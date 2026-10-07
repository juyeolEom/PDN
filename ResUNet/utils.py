import random

import numpy as np
import torch


def get_device(gpu_string):
    ids = get_device_ids(gpu_string)
    return torch.device(f'cuda:{ids[0]}' if ids else 'cpu')


def get_device_ids(gpu_string):
    if not torch.cuda.is_available():
        return []
    ids = [int(value.strip()) for value in gpu_string.split(',')]
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= torch.cuda.device_count() for i in ids):
        raise ValueError('GPU must list unique available CUDA device indices.')
    return ids


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id, base_seed=2026):
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def seed_noise_worker(worker_id):
    """Use DataLoader's epoch-specific seed instead of restarting each epoch."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def build_mask_generator(seed, device):
    """Create an RNG dedicated to random subsampling masks."""
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        # Fallback for older PyTorch versions that do not accept device=.
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
