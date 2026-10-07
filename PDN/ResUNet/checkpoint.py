"""Save final model weights and read new or legacy weights."""
from pathlib import Path
from datetime import datetime

import torch
from torch import nn


def unwrap_model(net):
    return net.module if isinstance(net, nn.DataParallel) else net


def load_weights(path, net, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    unwrap_model(net).load_state_dict(state)


def save_weights(net, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in unwrap_model(net).state_dict().items()}
    torch.save(state, path)


def create_run_directory(base_path):
    directory = Path(base_path) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    return directory
