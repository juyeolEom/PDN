"""Independent zero-mean Gaussian noise with one standard deviation per patch."""
import numpy as np


class RandomAdditiveNoise:
    def __init__(self, noise_min=0.0, noise_max=0.0):
        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)
        if not (np.isfinite(self.noise_min) and np.isfinite(self.noise_max)
                and 0 <= self.noise_min <= self.noise_max):
            raise ValueError('Require finite 0 <= noise_min <= noise_max.')
        self.enabled = self.noise_max > 0

    def __call__(self, patch):
        if not self.enabled:
            return patch
        sigma = np.random.uniform(self.noise_min, self.noise_max)
        noise = np.random.standard_normal(patch.shape).astype(patch.dtype)
        return patch + sigma * noise
