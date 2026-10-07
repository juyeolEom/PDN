# PDN (Pulse Denoising Network)

Self-supervised fluorescence pulse denoising with a residual 3D U-Net.

## Quick start

Run these commands from the repository directory. Supply your own `.npy` data;
sample data and pretrained weights are not included. Edit the paths in the JSON
configs before running.

```bash
# Install dependencies (see the CUDA note below).
python -m pip install -r requirements.txt

# Train from scratch.
python train.py --config configs/train.json

# Optional: Gaussian noise vaccination for 500 additional epochs.
# Set weight_load_path in augment.json to the weight saved above.
python train.py --config configs/augment.json

# Inference: set weight_load_path in test.json to the desired trained weight.
python train.py --config configs/test.json
```

Set `"st_mixing": true` in either training config to enable ST mixing, or leave
it `false` to use time-locked sampling. Each completed run saves its output and
the effective `config.json` together in a new timestamped directory.

## Installation

Use Python 3.9 or newer. Install PyTorch appropriate for your CPU/CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the following commands from this repository directory. Paths in configs are relative to the working directory. Example paths are placeholders; provide your own data and pretrained weights.

## Initial training

```bash
python train.py --config configs/train.json
```

The only data/model paths are `data_path` and `weight_save_path` (a base directory). Training starts from random weights. `n_epochs` defaults to 1500. Pretrained loading and Gaussian noise options are unavailable in this mode.

## Gaussian noise vaccination

```bash
python train.py --config configs/augment.json
```

Set `weight_load_path` to an existing weight file, plus `data_path`, `weight_save_path`, `noise_min`, and `noise_max`. This mode always trains for **500 additional epochs** with a new optimizer and scheduler. The epoch count cannot be overridden.

For each patch, sigma is drawn uniformly from `[noise_min, noise_max]`. Independent zero-mean Gaussian noise with that standard deviation is added to every time bin and pixel after mean subtraction, spatial transforms, and temporal rolling, before subsampling. Values are direct standard deviations in the input data's units; no Aref scaling, clipping, or re-centering is applied. Require `0 <= noise_min <= noise_max` and `noise_max > 0`. The example `noise_max=0.01` is only a placeholder, not a recommended level for every dataset.

## ST mixing

Both training configs accept a JSON boolean:

```json
"st_mixing": true
```

- `false` (default): choose an input/target neighbor triplet per 2 x 2 spatial block and share it across all time bins.
- `true`: choose the triplet independently for each spatial block and time bin (the original commented time-variant sampling).

This setting is independent of Gaussian augmentation. CLI overrides are `--st_mixing` and `--no-st_mixing`. No ST mixing is applied during inference.

## Inference

```bash
python train.py --config configs/test.json
```

The only data/model paths are `data_path` and `weight_load_path`. Results are saved next to the loaded weight, inside a new timestamped directory.

## Saved outputs

Every completed run creates a subdirectory using the execution machine's local date/time, including microseconds:

```text
weights/
  YYYYMMDD_HHMMSS_ffffff/
    weights.pth
    config.json
    YYYYMMDD_HHMMSS_ffffff/
      sample01_denoised.npy
      config.json
```

Training always saves the final model once to `weight_save_path/<timestamp>/weights.pth`. There are no best, intermediate, scratch, or optimizer checkpoints. During training, one loss line is printed per epoch (sum of batch losses); the saved path is printed after completion. No plots are generated.

Inference writes `<weight parent>/<timestamp>/<data stem>_denoised.npy`. It retains the full input shape. Existing output files are not overwritten. New weight files are plain model state dictionaries, independent of single/multiple GPU wrapping. Loading also accepts legacy dictionaries containing `model_state_dict`; any stored optimizer state is ignored.

Both training and inference save `config.json` beside their output. It contains
the merged defaults, input config, and CLI overrides, with the effective mask
seed filled in for training. The source config path is omitted. In augmentation
mode, `phase: "augment"` always means 500 additional epochs; no epoch override is
stored. Saved configs can be reused with `python train.py --config <saved-config>`
from the same working directory (relative paths keep their original meaning).

## Data and configuration

Input: a finite floating-point NumPy array of shape `(128, height, width)`, converted to float32. Training data must be at least `patch_x` in each spatial dimension. `patch_x` must be a multiple of 4 and at least 32.

Inference retains the original square overlapping-grid layout: both spatial dimensions must equal `test_grid * (patch_x / 2)`. Defaults (`patch_x=256`, `test_grid=8`) cover a 1024 x 1024 field. Output shape is inferred from the data; incompatible dimensions raise an error.

Configs include GPU indices, random seeds, patch/batch sizes, and optimizer/scheduler settings. `GPU` defaults to `"0"`; comma-separated indices enable multiple GPUs, and CPU is used when CUDA is unavailable. CLI options override the JSON values:

```bash
python train.py --config configs/train.json --n_epochs 100 --data_path data/sample02.npy
python train.py --phase augment --data_path data/sample01.npy --weight_load_path weights/pretrained.pth --weight_save_path weights --noise_min 0 --noise_max 0.01 --st_mixing
python train.py --phase test --help
```

Unknown or mode-inappropriate options are rejected. Test config has no training settings. Real data, weights, generated outputs, caches, and IDE settings are excluded by `.gitignore`.

## Checks

```bash
python -m unittest discover -s tests -v
```

Runtime checks require the dependencies above; otherwise those checks are reported as skipped. CLI/config checks use only the Python standard library.

## License

Copyright (c) 2026 KAIST BOOM Lab.

PDN is available under the [PolyForm Noncommercial License 1.0.0](LICENSE).
Noncommercial use, modification, and redistribution are permitted subject to
its terms. Commercial uses outside the license's permitted purposes require
separate permission from KAIST BOOM Lab.

The standard license expressly permits use by educational institutions, public
research organizations, and other organizations listed in its Noncommercial
Organizations section regardless of funding source or obligations resulting
from that funding. See the full license for the scope of these permissions.

When redistributing the software, include the license (or its official URL) and
the required copyright notice in [NOTICE](NOTICE). This is source-available
software with noncommercial restrictions.
