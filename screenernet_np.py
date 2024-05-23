import jax
import jax.numpy as jnp
from typing import Callable, Sequence, Any
from functools import partial

import flax

import optax

import tqdm

from networks import MixtureNeuralProcess, MLP, MeanAggregator, SequenceAggregator, NonLinearMVN, ResBlock

from networks import MLP

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

def screenernet_loss(screenernet, screenernet_input, apply_fn, losses):
    """
    Computes the objective loss of ScreenerNet.
    """
    weights = apply_fn(screenernet, screenernet_input).flatten()

    def body_fun(i, loss_sn):
        loss = losses[i]
        weight = weights[i]
        regularization_term = (1 - weight) * (1 - weight) * loss + weight * weight * jnp.maximum(1 - loss, 0)
        return loss_sn + regularization_term

    loss_screenernet = 0.0
    loss_screenernet = jax.lax.fori_loop(0, len(losses), body_fun, loss_screenernet)
    loss_screenernet = loss_screenernet * (1 / len(losses))
    return loss_screenernet


@partial(jax.jit, static_argnums=(0, 1, 2, 9, 10))
def np_losses_batch(apply_fn, elbo_fn, f_size, np_params, xs_context, ys_context, xs_target, ys_target,
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


@partial(jax.jit, static_argnums=(0, 1, 2, 10, 11))
def np_weighted_loss(apply_fn, elbo_fn, f_size, np_params, weights, xs_context, ys_context, xs_target,
                     ys_target, key, kl_penalty, num_posterior_mc):
    """
    Computes the weighted loss for a batch of tasks.
    """
    elbos = np_losses_batch(apply_fn, elbo_fn, f_size, np_params, xs_context, ys_context,
                            xs_target, ys_target, key, kl_penalty, num_posterior_mc)
    return -jnp.multiply(elbos, weights).mean()


@partial(jax.jit, static_argnums=(0, 1, 2, 11, 12, 13))
def update_np(
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

    value, grad = (jax.value_and_grad(np_weighted_loss, argnums=3)
                   (apply_fn, elbo_fn, f_size, theta, weights, xs_context, ys_context, xs_target, ys_target,
                    random_key, kl_penalty, num_posterior_mc))

    updates, opt_state = optimizer.update(grad, opt_state, theta)
    theta = optax.apply_updates(theta, updates)

    return theta, opt_state, value


def update_screenernet(tx, screenernet_opt, screenernet_input, apply_fn, screenernet, losses):
    """
    Performs one gradient step on the ScreenerNet.
    """
    loss_grad_fn = jax.value_and_grad(screenernet_loss, argnums=0)
    loss_val, grads = loss_grad_fn(screenernet, screenernet_input, apply_fn, losses)
    updates, opt_state = tx.update(grads, screenernet_opt)
    screenernet = optax.apply_updates(screenernet, updates)
    return loss_val, screenernet


def normalize_array(x, m, M):
    return (x - m) / (M - m)


def train(dataloader, dataset_size, context_size, num_epochs, rng, kl_penalty, num_posterior_mc):
    """
    Performs training of the NP and ScreenerNet.
    """
    key, rng = jax.random.split(rng)
    np_model, np_params = initialize_np(key, dataset_size)
    key, rng = jax.random.split(rng)
    sn_model = MLP([2 * context_size, 128, 128, 1], activation=jax.nn.sigmoid, activate_final=True, use_layernorm=True)
    dummy = jax.random.normal(key, (2 * context_size,))
    screenernet_params = sn_model.init(key, dummy)
    optimizer, opt_state = initialize_optimizer(np_params)
    tx = optax.adam(learning_rate=1e-3)
    sn_opt_state = tx.init(screenernet_params)
    data_iter = iter(dataloader)
    best, best_params = jnp.inf, np_params
    np_losses = list()
    for _ in (pbar := tqdm.trange(num_epochs, desc='Optimizing params. ')):
        batch = next(data_iter)
        batch = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch)
        xs_context, ys_context, xs_target, ys_target = batch
        screenernet_input = batch_to_screenernet_input(xs_context, ys_context)
        key, rng = jax.random.split(rng)
        losses = np_losses_batch(np_model.apply, np_model.elbo, 2 * context_size, np_params,
                                         xs_context, ys_context, xs_target, ys_target,
                                         key, kl_penalty=kl_penalty, num_posterior_mc=num_posterior_mc)
        # TODO: what should be their range and how should they be interpreted?
        loss_np = -losses.mean()
        losses = (losses - jnp.min(losses, axis=None)) / (jnp.max(losses, axis=None) - jnp.min(losses, axis=None)) # TODO: better normalization
        weights = sn_model.apply(screenernet_params, screenernet_input).flatten()
        sum_weights = jnp.sum(weights, axis=None)
        if sum_weights != 0:
          weights = (1 / sum_weights) * weights
        rng, key = jax.random.split(rng)
        np_params, opt_state, loss_np_weighted = update_np(np_model.apply, np_model.elbo, 2 * context_size, np_params, opt_state,
                                                  weights, xs_context, ys_context, xs_target, ys_target, key, optimizer,
                                                  kl_penalty, num_posterior_mc)
        loss_sn, screenernet_params = update_screenernet(tx, sn_opt_state, screenernet_input,
                                                         sn_model.apply, screenernet_params, losses)
        np_losses.append(loss_np)
        if loss_np < best:
            best = loss_np
            best_params = np_params
        pbar.set_description(f'Optimizing params. Losses: {loss_sn:.4f} {loss_np:.4f}')
    return np_model, best_params, np_losses
