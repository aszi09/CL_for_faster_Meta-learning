from typing import Callable, Sequence, Any
from functools import partial
import os

import time
import sys
import argparse
import jax
import jax.numpy as jnp

from torch.utils.data import Dataset, Sampler, DataLoader
import torch

import numpy as np

import flax
import flax.linen as nn

import optax
import pickle
from flax.training import train_state, orbax_utils

import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from functions import Fourier, Mixture, Slope, Polynomial, WhiteNoise, Shift
from networks import MixtureNeuralProcess, MLP, MeanAggregator, SequenceAggregator, NonLinearMVN, ResBlock

class ProbabilitySampler(Sampler):
    def __init__(self, probs):
        self.probs = probs / np.sum(probs)  # Ensure probabilities sum to 1

    def __iter__(self):
        # Sample indices according to the probabilities
        p = self.probs
        p = np.asarray(p).astype('float64')
        if p.sum() != 0:
            p = p * (1. / p.sum())
        indices = np.random.choice(len(self.probs), size=len(self.probs), replace=True, p=p)
        return iter(indices)

    def __len__(self):
        return len(self.probs)

    def update_probs(self, new_probs):
        self.probs = new_probs / np.sum(new_probs)  # Update and normalize probabilities


class SimpleDataset(Dataset):
    def __init__(self, dataset, context_size, dataset_size):
        self.context_size = context_size
        self.dataset_size = dataset_size
        self.context_xs, self.target_xs, self.context_ys, self.target_ys, self.distribs, self.noises = self._get_data(dataset)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.context_xs[idx], self.context_ys[idx], self.target_xs[idx], self.target_ys[idx], self.distribs[idx], self.noises[idx], idx

    def _get_data(self, dataset):
      xs_ys, distribs, noises = dataset
      xs, ys = xs_ys
      context_xs, target_xs = jnp.split(xs, indices_or_sections=(self.context_size, ), axis=1)
      context_ys, target_ys = jnp.split(ys, indices_or_sections=(self.context_size, ), axis=1)
      return context_xs, target_xs, context_ys, target_ys, distribs, noises
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
    return choice * (-2 - jnp.cos(2 * jnp.pi * x)) + (1 - choice) * (2 + jnp.cos(2 * jnp.pi * x)) + corrupt * noise


@jax.jit
def get_prob_score(variance, count):
    return jax.lax.cond(count >= 2, lambda _: (jnp.sqrt(variance + (variance ** 2 / (count - 1)))), lambda _: 0.0, 0)


@jax.jit
def get_prob_scores(variances, counts, eps=0.05):
    probs = jax.vmap(get_prob_score)(variances, counts)
    return probs + eps

def numpy_collate(batch):
    transposed_data = list(zip(*batch))
    xs_context = np.array(transposed_data[0])
    ys_context = np.array(transposed_data[1])
    xs_target = np.array(transposed_data[2])
    ys_target = np.array(transposed_data[3])
    distrib = np.array(transposed_data[4])
    noise = np.array(transposed_data[5])
    idx = np.array(transposed_data[6])
    return torch.tensor(xs_context), torch.tensor(ys_context), torch.tensor(xs_target), torch.tensor(ys_target), torch.tensor(distrib), torch.tensor(noise), torch.tensor(idx)

@jax.jit
def add_err(x, count, m, s):
    new_count = count + 1
    new_m = jax.lax.cond((count > 1), lambda _: (m + (x - m) / count), lambda _: x, 0)
    # new_m = (m + (x - m) / count) if count > 1 else x
    new_s = jax.lax.cond((count > 1), lambda _: (s + (x - m) * (x - new_m)), lambda _ : 0.0, 0)
    # new_s = (s + (x - m) * (x - new_m)) if count > 1 else 0.0
    variance = jax.lax.cond((count > 1), lambda _ :(new_s / (count - 1)), lambda _ : 0.0, 0)
    # variance = new_s / (count - 1) if count > 1 else 0.0
    return variance, new_count, new_m, new_s

@partial(jax.jit, static_argnums=(5,))
def update_vars(counts, ms, ss, vars, new_val, ind):
    count = counts[ind]
    m = ms[ind]
    s = ss[ind]
    variance, new_count, new_m, new_s = add_err(new_val, count, m, s)
    new_ms = ms.at[ind].set(new_m)
    new_ss = ss.at[ind].set(new_s)
    new_counts = counts.at[ind].set(new_count)
    new_vars = vars.at[ind].set(variance)
    return new_vars, new_ms, new_ss, new_counts

def initialize_np(rng, dataset_size, test_resolution=500):
    rng, key_data, key_test, key_x = jax.random.split(rng, 4)

    keys_data = jax.random.split(key_data, (dataset_size,))
    keys_test = jax.random.split(key_test, (test_resolution,))

    xs = jax.random.uniform(key_x, (dataset_size,)) * 2 - 1
    ys = jax.vmap(f)(keys_data, xs)
    embedding_xs = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)
    embedding_ys = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)
    embedding_both = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)

    projection_posterior = NonLinearMVN(
        MLP([128, 64], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True))

    # output_model = nn.Sequential([
    #     ResBlock(
    #         MLP([128, 128], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True),
    #     ),
    #     ResBlock(
    #         MLP([128, 128], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True),
    #     ),
    #     nn.Dense(2)
    # ])
    output_model = MLP([128, 128, 2], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True)
    projection_outputs = NonLinearMVN(output_model)

    posterior_aggregator = MeanAggregator(projection_posterior)

    model = MixtureNeuralProcess(
        embedding_xs, embedding_ys, embedding_both,
        posterior_aggregator,
        projection_outputs
    )

    rng, key1, key2 = jax.random.split(rng, 3)
    params = model.init({'params': key1, 'default': key2}, xs[:, None], ys[:, None], xs[:3, None])
    return model, params


@jax.jit
def batch_to_screenernet_input(xs, ys):
    xs = xs[:, :, 0]
    ys = ys[:, :, 0]
    return jnp.concatenate((xs, ys), axis=1)


def initialize_optimizer(params):
    optimizer = optax.chain(
        optax.clip(.1),
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=1e-6),
    )
    opt_state = optimizer.init(params)
    return optimizer, opt_state


@partial(jax.jit, static_argnums=(2, 5))
def screenernet_loss(screenernet, screenernet_input, apply_fn, losses, flattened, alpha=0.0001):
    """
    Computes the objective loss of ScreenerNet.
    """
    weights = apply_fn(screenernet, screenernet_input).flatten()

    def body_fun(i, loss_sn):
        loss = losses[i]
        weight = weights[i]  # what is the value?
        regularization_term = (1 - weight) * (1 - weight) * loss + weight * weight * jnp.maximum(1.5 - loss, 0)
        return loss_sn + regularization_term

    # flat_loss = jnp.sum(jnp.abs(flattened))
    loss_screenernet = 0.0
    loss_screenernet = jax.lax.fori_loop(0, len(losses), body_fun, loss_screenernet)
    # loss_screenernet = loss_screenernet * (1 / len(losses)) + alpha * flat_loss
    loss_screenernet = loss_screenernet * (1 / len(losses))
    return loss_screenernet


@partial(jax.jit, static_argnums=(0, 1, 2, 9, 10))
def np_losses_batch_elbo(apply_fn, elbo_fn, f_size, np_params, xs_context, ys_context, xs_target, ys_target,
                         key, kl_penalty, num_posterior_mc):
    """
    Computes the un-weighted ELBOs for all tasks in a batch.
    """
    # Compute ELBO over batch of datasets
    elbos = jax.vmap(partial(
        apply_fn,
        np_params,
        beta=kl_penalty, k=num_posterior_mc,
        method=elbo_fn
    ))(
        xs_context, ys_context, xs_target, ys_target, rngs={'default': jax.random.split(key, f_size)}
    )
    return elbos


@partial(jax.jit, static_argnums=(0, 7))
def np_losses_batch_gll(apply_fn, np_params, xs_context, ys_context, xs_target, ys_target, key, num_posterior_mc, batch_size):
    """
    Computes the un-weighted log likelihood loss for all tasks in a batch.
    """
    key_ll, key_app = jax.random.split(key)
    means_batch, stds_batch = jax.vmap(partial(
        apply_fn,
        np_params,
        k=num_posterior_mc
    ))(
        xs_context, ys_context, xs_target, rngs={'default': jax.random.split(key, batch_size)}
    )
    # keys = jax.random.split(key_ll, ys_target.shape)
    means_batch = jnp.reshape(means_batch, (means_batch.shape[0], means_batch.shape[1]))
    vs_batch = jnp.square(jnp.reshape(stds_batch, (stds_batch.shape[0], stds_batch.shape[1])))
    ys_target = jnp.reshape(ys_target, (ys_target.shape[0], ys_target.shape[1]))
    losses = jax.vmap(sample_gaussian_ll_loss, in_axes=(0, 0, 0))(ys_target, means_batch, vs_batch)
    return losses


@partial(jax.jit, static_argnums=(0, 8))
def np_weighted_loss_gll(apply_fn, np_params, weights, xs_context, ys_context, xs_target, ys_target, key,
                         num_posterior_mc):
    """
    Computes the weighted loss for a batch of tasks.
    """
    losses = np_losses_batch_gll(apply_fn, np_params, xs_context, ys_context, xs_target, ys_target, key,
                                 num_posterior_mc)
    # losses = losses - jnp.minimum(0, jnp.min(losses)) # remove
    weighted_losses = losses * weights
    return weighted_losses.mean()  # try just *


@jax.jit
def elementwise_gaussian_ll_loss(y, mean, std):
    eps = 1e-6
    v = std * std
    return jnp.log(jnp.maximum(v, eps)) + (y - mean) ** 2 / jnp.maximum(eps, v)


@jax.jit
def sample_gaussian_ll_loss(ys, means, stds):
    losses = jax.vmap(elementwise_gaussian_ll_loss, in_axes=(0, 0, 0))(ys, means, stds)
    res = 0.5 * jnp.mean(losses)
    return res


@partial(jax.jit, static_argnums=(0, 1, 2, 10, 11))
def np_weighted_loss_elbo(apply_fn, elbo_fn, f_size, np_params, weights, xs_context, ys_context, xs_target,
                          ys_target, key, kl_penalty, num_posterior_mc):
    """
    Computes the weighted loss for a batch of tasks.
    """
    elbos = np_losses_batch_elbo(apply_fn, elbo_fn, f_size, np_params, xs_context, ys_context,
                                 xs_target, ys_target, key, kl_penalty, num_posterior_mc)
    weighted_elbos = elbos * weights
    return -weighted_elbos.mean()  # try just *


@partial(jax.jit, static_argnums=(0, 1, 2, 11, 12, 13))
def update_np_elbo(
        apply_fn,
        elbo_fn,
        f_size,
        theta: flax.typing.VariableDict,
        opt_state: optax.OptState,
        weights,
        xs_context,
        ys_context,
        xs_target,
        ys_target,
        random_key: flax.typing.PRNGKey,
        optimizer,
        kl_penalty,
        num_posterior_mc
) -> tuple[flax.typing.VariableDict, optax.OptState, jax.Array]:
    # Implements a generic SGD Step

    value, grad = (jax.value_and_grad(np_weighted_loss_elbo, argnums=3)
                   (apply_fn, elbo_fn, f_size, theta, weights, xs_context, ys_context, xs_target, ys_target,
                    random_key, kl_penalty, num_posterior_mc))

    updates, opt_state = optimizer.update(grad, opt_state, theta)
    theta = optax.apply_updates(theta, updates)

    return theta, opt_state, value


@partial(jax.jit, static_argnums=(0, 9, 10))
def update_np_gll(
        apply_fn,
        theta: flax.typing.VariableDict,
        opt_state: optax.OptState,
        weights,
        xs_context,
        ys_context,
        xs_target,
        ys_target,
        random_key: flax.typing.PRNGKey,
        optimizer,
        num_posterior_mc
) -> tuple[flax.typing.VariableDict, optax.OptState, jax.Array]:
    # Implements a generic SGD Step
    value, grad = (jax.value_and_grad(np_weighted_loss_gll, argnums=1)
                   (apply_fn, theta, weights, xs_context, ys_context, xs_target, ys_target,
                    random_key, num_posterior_mc))
    updates, opt_state = optimizer.update(grad, opt_state, theta)
    theta = optax.apply_updates(theta, updates)

    return theta, opt_state, value


@partial(jax.jit, static_argnums=(0, 3))
def update_screenernet(tx, screenernet_opt, screenernet_input, apply_fn, screenernet, losses, vars):
    """
    Performs one gradient step on the ScreenerNet.
    """
    loss_grad_fn = jax.value_and_grad(screenernet_loss, argnums=0)
    loss_val, grads = loss_grad_fn(screenernet, screenernet_input, apply_fn, losses, vars)
    updates, opt_state = tx.update(grads, screenernet_opt)
    screenernet = optax.apply_updates(screenernet, updates)
    return loss_val, screenernet


@partial(jax.jit, static_argnums=(0,))
def evaluate(apply_fn, np_params, key, batch):
    X, y, x_test, y_test, distrib, noise, idx = batch
    X = X.reshape((X.shape[1], X.shape[0], X.shape[2]))
    y = y.reshape((y.shape[1], y.shape[0], y.shape[2]))
    x_test = x_test.reshape((x_test.shape[1], x_test.shape[0], x_test.shape[2]))
    y_test = y_test.reshape((y_test.shape[1], y_test.shape[0], y_test.shape[2]))
    # key_ll, key_eval = jax.random.split(key)
    means, stds = apply_fn(
        np_params,
        X[:, None], y[:, None], x_test[:, None],
        k=1,
        rngs={'default': key}
    )
    # keys = jax.random.split(key_ll, y_test.shape[0])
    L = sample_gaussian_ll_loss(y_test, means, stds)
    return L


"""
def screenernet_loss(screenernet, screenernet_input, apply_fn, losses, flattened, alpha=0.0001):

    weights = apply_fn(screenernet, screenernet_input).flatten()
    def body_fun(i, loss_sn):
        loss = losses[i]
        weight = weights[i] # what is the value?
        regularization_term = (1 - weight) * (1 - weight) * loss + weight * weight * jnp.maximum(1.5 - loss, 0)
        return loss_sn + regularization_term
    # flat_loss = jnp.sum(jnp.abs(flattened))
    loss_screenernet = 0.0
    loss_screenernet = jax.lax.fori_loop(0, len(losses), body_fun, loss_screenernet)
    # loss_screenernet = loss_screenernet * (1 / len(losses)) + alpha * flat_loss
    loss_screenernet = loss_screenernet * (1 / len(losses))
    return loss_screenernet
    """


def batch_update_errs(losses, indices, ms, ss, counts, all_vars):
    indices_jnp = jnp.array(indices)

    def body_fun(i, agg):
        err = losses[i]
        sample_ind = indices_jnp[i]
        agg_vars, agg_ms, agg_ss, agg_counts = agg
        return update_vars(agg_counts, agg_ms, agg_ss, agg_vars, err, sample_ind)

    return jax.lax.fori_loop(0, len(losses), body_fun, (all_vars, ms, ss, counts))


def train(train_dataset, test_dataset, dataset_size, context_size, num_epochs, rng, kl_penalty, num_posterior_mc, batch_size):
    """
    Performs training of the NP and ScreenerNet.
    """
    key, rng = jax.random.split(rng)
    np_model, np_params = initialize_np(key, dataset_size)
    key, rng = jax.random.split(rng)
    # sn_model = MLP([2 * context_size, 128], activation=jax.nn.sigmoid, activate_final=False, use_layernorm=True)
    sn_model = nn.Sequential([
        MLP([2 * context_size, 64, 64, 128], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True),
        MLP([128, 1], activation=jax.nn.relu, activate_final=True, use_layernorm=False)
    ])
    dummy = jax.random.normal(key, (2 * context_size,))
    screenernet_params = sn_model.init(key, dummy)
    optimizer, opt_state = initialize_optimizer(np_params)
    tx = optax.adam(learning_rate=1e-3)
    sn_opt_state = tx.init(screenernet_params)
    best, best_params = jnp.inf, np_params
    np_sn_losses = list()
    baseline_losses = list()
    screenernet_losses = list()
    key, rng = jax.random.split(rng)
    baseline_model, baseline_params = initialize_np(key, dataset_size)
    baseline_optimizer, baseline_opt_state = initialize_optimizer(baseline_params)
    best_baseline, best_baseline_params = jnp.inf, baseline_params
    for epoch in (pbar := tqdm.trange(num_epochs, desc='Optimizing params. ')):
        dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=numpy_collate)
        test_dl = DataLoader(test_dataset, shuffle=True, batch_size=1, collate_fn=numpy_collate)
        data_it = iter(dl)
        for stp in range(int(dataset_size / batch_size)):
            batch = next(data_it)
            batch = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch)
            xs_context, ys_context, xs_target, ys_target, distrib, noise, idx = batch
            screenernet_input = batch_to_screenernet_input(xs_context, ys_context)
            key, rng = jax.random.split(rng)
            losses = np_losses_batch_elbo(np_model.apply, np_model.err, batch_size, np_params, xs_context, ys_context,
                                          xs_target, ys_target, key, kl_penalty, num_posterior_mc)  # z-score
            # print(losses)
            loss_np = losses.mean()
            weights = sn_model.apply(screenernet_params, screenernet_input).flatten()
            if epoch < num_epochs / 4:
                weights = jnp.ones(weights.shape)
            sum_weights = jnp.sum(weights, axis=None)
            if sum_weights != 0:
                weights = (batch_size / sum_weights) * weights  # remove
            rng, key = jax.random.split(rng)
            np_params, opt_state, loss_np_weighted = update_np_elbo(np_model.apply, np_model.elbo, batch_size,
                                                                    np_params, opt_state, weights, xs_context,
                                                                    ys_context, xs_target, ys_target, key, optimizer,
                                                                    kl_penalty, num_posterior_mc)  # elbo
            vars = jnp.concatenate([arr.flatten() for arr in jax.tree_util.tree_leaves(screenernet_params["params"])])
            flattened = jnp.array(vars)
            loss_sn, screenernet_params = update_screenernet(tx, sn_opt_state, screenernet_input,
                                                             sn_model.apply, screenernet_params, losses, flattened)
            rng, key = jax.random.split(rng)
            baseline_params, baseline_opt_state, baseline_loss = update_np_elbo(baseline_model.apply,
                                                                                baseline_model.elbo, batch_size,
                                                                                baseline_params, baseline_opt_state,
                                                                                jnp.ones(batch_size), xs_context,
                                                                                ys_context, xs_target, ys_target, key,
                                                                                baseline_optimizer, kl_penalty,
                                                                                num_posterior_mc)
            if loss_np_weighted < best:
                best = loss_np_weighted
                best_params = np_params
            if baseline_loss < best_baseline:
                best_baseline = baseline_loss
                best_baseline_params = baseline_params
            screenernet_losses.append(loss_sn)
            pbar.set_description(
                f'Optimizing params. Losses: {loss_sn:.4f} {loss_np:.4f} {baseline_loss: .4f} {loss_np_weighted: .4f}')
        # if epoch % 10 == 9:
        key1, key2, rng = jax.random.split(rng, 3)
        batch1 = next(iter(test_dl))
        batch2 = next(iter(test_dl))
        batch1 = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch1)
        batch2 = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch2)
        loss_sn_np = evaluate(np_model.apply, best_params, key1, batch1)
        loss_baseline = evaluate(baseline_model.apply, best_baseline_params, key2, batch2)
        np_sn_losses.append(loss_sn_np)
        baseline_losses.append(loss_baseline)
    return np_model, best_params, best_baseline_params, screenernet_losses, np_sn_losses, baseline_losses


def checkpoint_save(np_params, baseline_params, losses_ab_train, losses_ab_eval, losses_baseline_train,
                    losses_baseline_eval, path, name):
    os.makedirs(path, exist_ok=True)
    pickle_dict = {
        'np_params': np_params,
        'baseline_params': baseline_params,
        'losses_ab_train': losses_ab_train,
        'losses_ab_eval': losses_ab_eval,
        'losses_baseline_train': losses_baseline_train,
        'losses_baseline_eval': losses_baseline_eval
    }
    with open(os.path.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(pickle_dict, f)


def train_active_bias(train_dataset, test_dataset, dataset_size, context_size, num_epochs, rng, kl_penalty,
                      num_posterior_mc, np_model, np_params, optimizer, opt_state, path, batch_size):
    """
    Performs training of Neural Processes using Active Bias sampling.
    """
    ms = jnp.zeros(dataset_size)
    ss = jnp.zeros(dataset_size)
    counts = jnp.zeros(dataset_size)
    all_vars = jnp.zeros(dataset_size)
    key, rng = jax.random.split(rng)
    key, rng = jax.random.split(rng)
    best, best_params = jnp.inf, np_params
    key, rng = jax.random.split(rng)
    baseline_model, baseline_params = initialize_np(key, dataset_size)
    baseline_optimizer, baseline_opt_state = initialize_optimizer(baseline_params)
    best_baseline, best_baseline_params = jnp.inf, baseline_params
    batch_sampler = ProbabilitySampler(jnp.ones(dataset_size))
    losses_ab_train = list()
    losses_ab_eval = list()
    losses_baseline_train = list()
    losses_baseline_eval = list()
    for epoch in (pbar := tqdm.trange(num_epochs, desc='Optimizing params. ')):
        if epoch >= int(num_epochs * 0.10):
            key, rng = jax.random.split(rng)
            scores = get_prob_scores(all_vars, counts)  # what is the window size?
            batch_sampler.update_probs(scores)
        dl_ab = DataLoader(train_dataset, sampler=batch_sampler, batch_size=batch_size, collate_fn=numpy_collate)
        dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=numpy_collate)
        test_dl = DataLoader(test_dataset, shuffle=True, batch_size=1, collate_fn=numpy_collate)
        data_it = iter(dl)
        data_it_ab = iter(dl_ab)
        for stp in range(int(dataset_size / batch_size)):
            batch = next(data_it)
            batch_ab = next(data_it_ab)
            batch = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch)
            batch_ab = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch_ab)
            xs_context, ys_context, xs_target, ys_target, distrib, noise, idx = batch
            xs_context_ab, ys_context_ab, xs_target_ab, ys_target_ab, distrib_ab, noise_ab, idx_ab = batch_ab
            key, rng = jax.random.split(rng)
            losses = np_losses_batch_elbo(np_model.apply, np_model.elbo, batch_size, np_params, xs_context, ys_context,
                                          xs_target, ys_target, key, kl_penalty, num_posterior_mc)
            all_vars, ms, ss, counts = batch_update_errs(losses, idx_ab, ms, ss, counts, all_vars)
            weights = jnp.ones(batch_size, dtype=jnp.float32)
            rng, key = jax.random.split(rng)
            np_params, opt_state, loss_np = update_np_elbo(np_model.apply, np_model.elbo, batch_size, np_params,
                                                           opt_state, weights, xs_context, ys_context, xs_target,
                                                           ys_target, key, optimizer, kl_penalty,
                                                           num_posterior_mc)  # elbo
            rng, key = jax.random.split(rng)
            baseline_params, baseline_opt_state, baseline_loss = update_np_elbo(baseline_model.apply,
                                                                                baseline_model.elbo, batch_size,
                                                                                baseline_params, baseline_opt_state,
                                                                                jnp.ones(batch_size), xs_context,
                                                                                ys_context, xs_target, ys_target, key,
                                                                                baseline_optimizer, kl_penalty,
                                                                                num_posterior_mc)
            if loss_np < best:
                best = loss_np
                best_params = np_params
            if baseline_loss < best_baseline:
                best_baseline = baseline_loss
                best_baseline_params = baseline_params
            losses_ab_train.append(loss_np)
            losses_baseline_train.append(baseline_loss)
            pbar.set_description(f'Optimizing params. Losses: {loss_np:.4f} {baseline_loss: .4f}')
        if epoch % 5 == 4:
            key1, key2, rng = jax.random.split(rng, 3)
            batch1 = next(iter(test_dl))
            batch2 = next(iter(test_dl))
            batch1 = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch1)
            batch2 = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch2)
            loss_ab = evaluate(np_model.apply, best_params, key1, batch1).item()
            loss_baseline = evaluate(baseline_model.apply, best_baseline_params, key2, batch2).item()
            losses_ab_eval.append(loss_ab)
            losses_baseline_eval.append(loss_baseline)
            checkpoint_save(np_params, baseline_params, losses_ab_train, losses_ab_eval, losses_baseline_train,
                            losses_baseline_eval, path, 'aaa')
    return np_model, best_params, best_baseline_params, losses_ab_train, losses_ab_eval, losses_baseline_train, losses_baseline_eval


def simple_train(random_seed, path, sampler_ratio_train, sampler_ratio_test, noise_levels):
    batch_size = 128
    context_size = 64
    target_size = 32
    num_epochs = 50
    kl_penalty = 1e-4
    num_posterior_mc = 1
    rng = jax.random.key(0)
    test_resolution = 512
    dataset_size = 128 * 100
    key, rng = jax.random.split(rng)
    dataset_train, dataset_test = generate_datasets(sampler_ratio_train, sampler_ratio_test, dataset_size, batch_size, target_size, context_size, key, noise_levels)
    rng = jax.random.key(random_seed)
    key, rng = jax.random.split(rng)
    np_model, np_params = initialize_np(key, dataset_size)
    optimizer, opt_state = initialize_optimizer(np_params)
    np_model, best_params, best_baseline_params, losses_ab_train, losses_ab_eval, losses_baseline_train, losses_baseline_eval = train_active_bias(
        dataset_train, dataset_test, dataset_size, context_size, num_epochs, rng, kl_penalty, num_posterior_mc,
        np_model, np_params, optimizer, opt_state, path, batch_size)
    print(losses_ab_train)


def generate_datasets(sampler_ratio_train, sampler_ratio_test, dataset_size, batch_size, target_size, context_size, key, noise_levels):
    key_test, key_train = jax.random.split(key)
    f2 = Fourier(n=2, amplitude=.5, period=1.0)
    f5 = Slope()
    f6 = Polynomial(order=2, clip_bounds=(-1, 1))
    FOURIER = 0
    POLYN = 1
    SLOPE = 2

    data_sampler1 = partial(
        joint,
        WhiteNoise(f2, noise_levels[0]),
        partial(uniform, n=context_size + target_size, bounds=(-1, 1))
    )
    data_sampler_props_1 = {
        "distribution": FOURIER,
        "noise": noise_levels[0],
        "sampler": data_sampler1
    }
    data_sampler2 = partial(
        joint,
        WhiteNoise(f5, noise_levels[1]),
        partial(uniform, n=context_size + target_size, bounds=(-1, 1))
    )
    data_sampler_props_2 = {
        "distribution": SLOPE,
        "noise": noise_levels[1],
        "sampler": data_sampler2
    }
    data_sampler3 = partial(
        joint,
        WhiteNoise(f6, noise_levels[2]),
        partial(uniform, n=context_size + target_size, bounds=(-1, 1))
    )
    data_sampler_props_3 = {
        "distribution": POLYN,
        "noise": noise_levels[2],
        "sampler": data_sampler3
    }
    dataset_train = SimpleDataset(
        generate_noisy_split_trainingdata([data_sampler_props_1, data_sampler_props_2, data_sampler_props_3],
                                          sampler_ratio_train, dataset_size, key_train), context_size=context_size,
        dataset_size=dataset_size)
    dataset_test = SimpleDataset(
        generate_noisy_split_trainingdata([data_sampler_props_1, data_sampler_props_2, data_sampler_props_3],
                                          sampler_ratio_test, batch_size * 22, key_test), context_size=context_size,
        dataset_size=batch_size * 22)
    return dataset_train, dataset_test
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

    params = module.init({'params': key_param, 'default': key_rng}, jnp.zeros(()))
    xs, ys = data_sampler(module, params, key_data)

    if return_params:
        return xs, ys, params
    return xs, ys


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


@partial(jax.jit, static_argnums=(1))
def gen_sampler_datapoint(key, sampler):
    x, y = sampler(key)
    x, y = x[..., None], y[..., None]
    return x, y


@partial(jax.jit, static_argnums=(1,2))
def generate_dataset(rng, num_batches, sampler):
    keys = jax.random.split(rng, num_batches)
    batched_generate = jax.vmap(partial(gen_sampler_datapoint, sampler=sampler))
    x, y = batched_generate(keys)
    return x, y


def generate_noisy_split_trainingdata(samplers, sampler_ratios, dataset_size, rng):
    """
    Generate a dataset with a split of different samplers and ratios
    """

    assert len(samplers) == len(sampler_ratios), "The number of samplers and ratios must be the same"
    assert sum(sampler_ratios) == 1.0, "The sum of the ratios must be 1.0"
    keys = jax.random.split(rng, len(samplers))
    datasets = []
    distribs = []
    noises = []
    for (sampler_prop, ratio, key) in zip(samplers, sampler_ratios, keys):
        sampler, distrib, noise = sampler_prop["sampler"], sampler_prop["distribution"], sampler_prop["noise"]
        dataset = generate_dataset(key, int(dataset_size*ratio), sampler)
        datasets.append(np.asarray(dataset))
        distribs.append(jnp.repeat(distrib, int(dataset_size*ratio)))
        noises.append(jnp.repeat(noise, int(dataset_size*ratio)))
    x_datasets, y_datasets = zip(*datasets)
    return  np.asarray((jnp.concatenate(x_datasets), jnp.concatenate(y_datasets))), jnp.concatenate(distribs), jnp.concatenate(noises)


def main():
    parser = argparse.ArgumentParser(description="Parse command line arguments into specified values.")

    parser.add_argument("function_name", type=str, help="The name of the function to be called.")
    parser.add_argument("--seed", type=int, required=True, help="An integer value for the random seed.")
    parser.add_argument("--path", type=str, required=True, help="A string value for the path.")
    parser.add_argument("--train_ratio", type=float, nargs=3, required=True, help="An array of 3 decimal numbers for sampler ratio train.")
    parser.add_argument("--test_ratio", type=float, nargs=3, required=True, help="An array of 3 decimal numbers for sampler ratio test.")
    parser.add_argument("--noise", type=float, nargs=3, required=True, help="An array of 3 decimal numbers for noise levels.")

    args = parser.parse_args()

    # Extracting the arguments into the specified variables
    random_seed = args.random_seed
    path = args.path
    sampler_ratio_train = args.sampler_ratio_train
    sampler_ratio_test = args.sampler_ratio_test
    noise_levels = args.noise_levels

    # Assuming the function is defined in the global namespace and we want to call it
    if args.function_name in globals():
        globals()[args.function_name](random_seed, path, sampler_ratio_train, sampler_ratio_test, noise_levels)
    else:
        print(f"Function {args.function_name} is not defined.")

if __name__ == "__main__":
    main()

