import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train import parse_args
from ResUNet.config import save_config

try:
    import numpy as np
    import torch
    from ResUNet import engine
    from ResUNet.additive_noise import RandomAdditiveNoise
    from ResUNet.checkpoint import load_weights, save_weights
    from ResUNet.sampling import generate_mask_pair
except ModuleNotFoundError:
    torch = None


class ConfigTests(unittest.TestCase):
    def test_saved_config_records_overrides_and_can_be_reused(self):
        root = Path(__file__).resolve().parents[1]
        for phase in ('train', 'augment', 'test'):
            args = ['--config', str(root / 'configs' / (phase + '.json')),
                    '--data_path', 'data/override.npy']
            if phase != 'test':
                args += ['--st_mixing', '--seed', '42']
            opt = parse_args(args)
            with tempfile.TemporaryDirectory() as tmp:
                saved_path = save_config(opt, tmp)
                saved = json.loads(saved_path.read_text(encoding='utf-8'))
                self.assertNotIn('config', saved)
                self.assertEqual(saved['data_path'], 'data/override.npy')
                if phase != 'test':
                    self.assertTrue(saved['st_mixing'])
                    self.assertEqual(saved['mask_seed'], 42)
                    self.assertIsNone(opt.mask_seed)
                replayed = parse_args(['--config', str(saved_path)])
                replayed_settings = vars(replayed).copy()
                replayed_settings.pop('config')
                self.assertEqual(replayed_settings, saved)

    def test_examples_and_overrides(self):
        root = Path(__file__).resolve().parents[1]
        for mode in ('train', 'augment', 'test'):
            opt = parse_args(['--config', str(root / 'configs' / (mode + '.json'))])
            self.assertEqual(opt.phase, mode)
        opt = parse_args(['--config', str(root / 'configs/train.json'), '--st_mixing', '--n_epochs', '2'])
        self.assertTrue(opt.st_mixing)
        self.assertEqual(opt.n_epochs, 2)

    def test_reject_incompatible_options(self):
        cases = [
            ['--phase', 'train', '--weight_load_path', 'x.pth'],
            ['--phase', 'train', '--noise_max', '0.1'],
            ['--phase', 'augment', '--noise_max', '0.1', '--n_epochs', '1'],
            ['--phase', 'augment', '--noise_max', '0'],
            ['--phase', 'test', '--weight_save_path', 'weights'],
            ['--phase', 'test', '--st_mixing'],
            ['--save_best', 'True'],
        ]
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)


@unittest.skipIf(torch is None, 'Install torch, numpy, and einops for runtime checks')
class RuntimeTests(unittest.TestCase):
    def test_temporal_masks(self):
        data = torch.zeros(1, 1, 128, 8, 8)
        for mixing in (False, True):
            masks = generate_mask_pair(data, torch.Generator().manual_seed(10), mixing)
            for mask in masks:
                blocks = mask.reshape(1, 128, 4, 4, 4)
                self.assertTrue(torch.all(blocks.sum(-1) == 1))
                self.assertEqual(bool(torch.equal(blocks[:, :1].expand_as(blocks), blocks)), not mixing)
            self.assertFalse(bool(torch.any(masks[0] & masks[1])))
            self.assertFalse(bool(torch.any(masks[0] & masks[2])))

    def test_noise_uses_direct_sigma(self):
        np.random.seed(42)
        noise = RandomAdditiveNoise(0.2, 0.2)(np.zeros((128, 64, 64), dtype=np.float32))
        self.assertAlmostEqual(float(noise.std()), 0.2, delta=0.002)
        self.assertAlmostEqual(float(noise.mean()), 0, delta=0.002)

    def test_weights_and_legacy_loading(self):
        net = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pth'
            save_weights(net, path)
            restored = torch.nn.Linear(2, 1)
            load_weights(path, restored, 'cpu')
            self.assertTrue(torch.equal(net.weight, restored.weight))
            torch.save({'model_state_dict': {'module.' + k: v for k, v in net.state_dict().items()}, 'optimizer_state_dict': {}}, path)
            load_weights(path, restored, 'cpu')
            self.assertTrue(torch.equal(net.weight, restored.weight))

    def test_inference_overlap_and_timestamped_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / 'sample01.npy'
            data = np.random.default_rng(42).normal(size=(128, 32, 32)).astype(np.float32)
            np.save(data_path, data)
            opt = parse_args(['--phase', 'test', '--data_path', str(data_path),
                              '--weight_load_path', str(root / 'weights.pth'),
                              '--patch_x', '32', '--test_grid', '2'])
            with patch.object(engine, 'load_weights'), contextlib.redirect_stdout(io.StringIO()):
                engine.test_epoch(torch.nn.Identity(), opt, torch.device('cpu'))
            outputs = list(root.glob('*/sample01_denoised.npy'))
            self.assertEqual(len(outputs), 1)
            self.assertTrue((outputs[0].parent / 'config.json').is_file())
            np.testing.assert_allclose(np.load(outputs[0]), data, atol=1e-6)

    def test_training_modes_save_once_at_end(self):
        for phase, epochs in [('train', 2), ('augment', 500)]:
            args = ['--phase', phase]
            args += ['--n_epochs', '2'] if phase == 'train' else ['--noise_max', '0.1']
            opt = parse_args(args)
            with tempfile.TemporaryDirectory() as tmp:
                opt.weight_save_path = tmp
                with patch.object(engine, 'build_model', return_value=torch.nn.Linear(2, 1)), patch.object(engine, 'build_trainloader', return_value=[]), patch.object(engine, 'train_epoch', return_value=1.0) as epoch, patch.object(engine, 'load_weights') as load, patch.object(engine, 'save_weights') as save, contextlib.redirect_stdout(io.StringIO()):
                    engine.run_train(opt)
                self.assertEqual(epoch.call_count, epochs)
                self.assertEqual(load.call_count, int(phase == 'augment'))
                save.assert_called_once()
                self.assertEqual(save.call_args.args[1].name, 'weights.pth')
                self.assertNotEqual(save.call_args.args[1].parent, Path(tmp))
                self.assertTrue((save.call_args.args[1].parent / 'config.json').is_file())


if __name__ == '__main__':
    unittest.main()
