"""PDN entry point: initial training, Gaussian augmentation, or inference."""
import argparse
import json
import math
from pathlib import Path


def build_parser(phase):
    parser = argparse.ArgumentParser(description='PDN pulse denoising')
    parser.add_argument('--phase', choices=['train', 'augment', 'test'], default=phase)
    parser.add_argument('--config', help='JSON configuration; CLI arguments override its values')
    parser.add_argument('--data_path', default='data/sample01.npy')
    parser.add_argument('--GPU', default='0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--patch_x', type=int, default=256)
    if phase == 'test':
        parser.add_argument('--weight_load_path', default='weights/pretrained.pth')
        parser.add_argument('--test_grid', type=int, default=8)
    else:
        parser.add_argument('--weight_save_path', default='weights', help='Base directory for timestamped runs')
        parser.add_argument('--st_mixing', action=argparse.BooleanOptionalAction, default=False,
                            help='Independent spatial sampling per time bin')
        parser.add_argument('--mask_seed', type=int, default=None)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--train_dataset_size', type=int, default=16)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--b1', type=float, default=0.5)
        parser.add_argument('--b2', type=float, default=0.99)
        parser.add_argument('--lr_decay', type=float, default=0.2)
        parser.add_argument('--lr_decay_patience', type=int, default=100)
        if phase == 'augment':
            parser.add_argument('--weight_load_path', default='weights/pretrained.pth')
            parser.add_argument('--noise_min', type=float, default=0.0)
            parser.add_argument('--noise_max', type=float, default=None)
        else:
            parser.add_argument('--n_epochs', type=int, default=1500)
    return parser


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config')
    bootstrap.add_argument('--phase', choices=['train', 'augment', 'test'])
    known, _ = bootstrap.parse_known_args(argv)
    settings = {}
    if known.config:
        try:
            settings = json.loads(Path(known.config).read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            bootstrap.error(str(exc))
        if not isinstance(settings, dict):
            bootstrap.error('Config must be a JSON object.')
    phase = known.phase or settings.get('phase', 'train')
    if phase not in ('train', 'augment', 'test'):
        bootstrap.error('Invalid phase in config.')
    parser = build_parser(phase)
    actions = {a.dest: a for a in parser._actions}
    unknown = set(settings) - (set(actions) - {'help', 'config'})
    if unknown:
        parser.error(f'Unsupported config options for {phase}: {sorted(unknown)}')
    for key, value in settings.items():
        action = actions[key]
        if key == 'st_mixing':
            if not isinstance(value, bool):
                parser.error('st_mixing must be true or false.')
            continue
        if value is None and key == 'mask_seed':
            continue
        expected = action.type or str
        if isinstance(value, bool) or not isinstance(value, (int, float) if expected is float else expected):
            parser.error(f'Invalid config type for {key}.')
    parser.set_defaults(**settings)
    opt = parser.parse_args(argv)
    for name in ('data_path', 'weight_save_path', 'weight_load_path', 'GPU'):
        if hasattr(opt, name) and not getattr(opt, name).strip():
            parser.error(f'{name} cannot be empty.')
    if opt.patch_x < 32 or opt.patch_x % 4:
        parser.error('patch_x must be a multiple of 4 and at least 32.')
    if not 0 <= opt.seed < 2**32:
        parser.error('seed must be between 0 and 2**32 - 1.')
    if phase == 'test':
        if opt.test_grid < 1:
            parser.error('test_grid must be positive.')
    else:
        if min(opt.batch_size, opt.train_dataset_size) < 1 or opt.num_workers < 0:
            parser.error('batch_size/train_dataset_size must be positive; num_workers must be nonnegative.')
        if not math.isfinite(opt.lr) or opt.lr <= 0 or not (0 <= opt.b1 < 1 and 0 <= opt.b2 < 1):
            parser.error('Require finite lr > 0 and 0 <= b1, b2 < 1.')
        if not 0 < opt.lr_decay < 1 or opt.lr_decay_patience < 0:
            parser.error('Require 0 < lr_decay < 1 and nonnegative patience.')
        if opt.mask_seed is not None and not 0 <= opt.mask_seed < 2**32:
            parser.error('mask_seed must be between 0 and 2**32 - 1.')
        if phase == 'train' and opt.n_epochs < 1:
            parser.error('n_epochs must be positive.')
        if phase == 'augment' and (opt.noise_max is None or not math.isfinite(opt.noise_min)
                or not math.isfinite(opt.noise_max) or not 0 <= opt.noise_min <= opt.noise_max
                or opt.noise_max <= 0):
            parser.error('Augmentation requires finite 0 <= noise_min <= noise_max and noise_max > 0.')
    return opt


def main():
    opt = parse_args()
    from ResUNet.engine import run_test, run_train
    if opt.phase == 'test':
        run_test(opt)
    else:
        run_train(opt)


if __name__ == '__main__':
    main()
