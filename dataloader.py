from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial

import torch
from jax import jit
from torch.utils.data import Dataset
import numpy as np
from jax.tree_util import tree_map
from torch.utils import data


from functions import Fourier, Mixture, Shift, WhiteNoise


class MixtureDataset(Dataset):
    def __init__(self, dataset_size, key, num_context_samples, num_target_samples):
        self.key = key
        self.dataset_size = dataset_size
        self.xs, self.ys = self._get_data()
        self.num_context_samples = num_context_samples
        self.num_target_samples = num_target_samples

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def _get_data(self):
        f1 = Fourier(n=4, amplitude=.5, period=1.0)
        f2 = Fourier(n=2, amplitude=.5, period=1.0)
        f3 = Fourier(n=6, amplitude=.5, period=2.0)
        f4 = Fourier(n=3, amplitude=1.0, period=2.0)

        def f(
                key: flax.typing.PRNGKey,
                x: jax.Array,
                noise_scale: float = 0.2,
                mixture_prob: float = 0.5,
                corrupt: bool = True
        ):
            key_noise, key_mixture = jax.random.split(key)

            noise = jax.random.normal(key, x.shape) * noise_scale
            choice = jax.random.bernoulli(key_mixture, mixture_prob, x.shape)

            # return choice * (jnp.sin(2 * jnp.pi * x / 2)) + (1 - choice) * (jnp.cos(2 * jnp.pi * 2 * x)) + corrupt * noise
            return choice * (-2 - jnp.cos(2 * jnp.pi * x)) + (1 - choice) * (
                        2 + jnp.cos(2 * jnp.pi * x)) + corrupt * noise

        def uniform(
                module: nn.Module,
                params: flax.typing.VariableDict,
                key: flax.typing.PRNGKey,
                n: int,
                bounds: tuple[float, float]
        ) -> tuple[jax.Array, jax.Array]:
            # Samples from p(X, Y | Z) = p(Y | Z, X)p(X)
            key_xs, key_ys = jax.random.split(key)
            xs = jax.random.uniform(key_xs, (n,)) * (bounds[1] - bounds[0]) + bounds[0]

            ys = jax.vmap(module.apply, in_axes=(None, 0))(params, xs, rngs={'default': jax.random.split(key_ys, n)})

            return xs, ys

        def joint(
                module: nn.Module,
                data_sampler: Callable[
                    [nn.Module, flax.typing.VariableDict, flax.typing.PRNGKey],
                    tuple[jax.Array, jax.Array]
                ],
                key: flax.typing.PRNGKey,
                return_params: bool = False
        ) -> tuple[jax.Array, jax.Array]:
            # Samples from p(Z, X, Y)
            key_param, key_rng, key_data = jax.random.split(key, 3)

            params = module.init({'param': key_param, 'default': key_rng}, jnp.zeros(()))
            xs, ys = data_sampler(module, params, key_data)

            if return_params:
                return xs, ys, params
            return xs, ys

        data_sampler = partial(
            joint,
            Mixture([WhiteNoise(Shift(f2, y_shift=1.0), 0.05), WhiteNoise(Shift(f4, y_shift=-1.0), 0.2)]),
            partial(uniform, n=(self.num_target_samples + self.num_context_samples), bounds=(-1, 1))
        )

        key_data, self.key = jax.random.split(self.key)
        xs, ys = jax.vmap(data_sampler)(jax.random.split(key_data, num=self.dataset_size))
        xs, ys = xs[..., None], ys[..., None]
        return xs, ys
