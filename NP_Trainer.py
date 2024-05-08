import os
import time
import jax.numpy as jnp
from typing import Callable, Sequence, Any
from functools import partial
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import jaxopt
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from functions import Fourier, Mixture, Slope, Polynomial, WhiteNoise, Shift
from networks import MixtureNeuralProcess, MLP, MeanAggregator, SequenceAggregator, NonLinearMVN, ResBlock
from dataloader import MixtureDataset

class NP_Trainer():
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 algorithm_name, data_prepare, model_prepare, data_curriculum,
                 model_curriculum):
        # self.random_seed = random_seed
        # set_random(self.random_seed)
        #
        # self.data_prepare = data_prepare
        # self.model_prepare = model_prepare
        # self.data_curriculum = data_curriculum
        # self.model_curriculum = model_curriculum
        # self.loss_curriculum = loss_curriculum
        #
        # self._init_dataloader(data_name)
        # self._init_model(data_name, net_name, device_name, num_epochs)
        # self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)
        self.rng = jax.random.key(0)
        self.num_posterior_mc = 10  # number of latents to sample from p(Z | X, Y)
        self.batch_size = 256  # number of functions to sample from p(Z)

        self.kl_penalty = 1e-4  # Note to self: magnitude of the kl-divergence can take over in the loss
        self.num_target_samples = 128
        self.num_context_samples = 256
        self.epochs = 10000

    def _init_dataloader(self, data_name):
        # train_dataset, valid_dataset, test_dataset = \
        #     get_dataset_with_noise('./data', data_name)
        #
        # self.train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=100, shuffle=True, num_workers=2, pin_memory=True)
        # self.valid_loader = torch.utils.data.DataLoader(
        #     valid_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
        # self.test_loader = torch.utils.data.DataLoader(
        #     test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
        #
        # self.data_prepare(self.train_loader)
        self.dataset = MixtureDataset
        self.loader = DataLoader(dataset=self.dataset, )

    def _init_optimizer(self):
        optimizer = optax.chain(
            optax.clip(.1),
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=1e-3, weight_decay=1e-6),
        )
        self.opt_state = optimizer.init(self.params)

    def _init_model(self, data_name, net_name, num_epochs):
        self.epochs = num_epochs
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=self.epochs, eta_min=1e-6)
        #
        # self.model_prepare(
        #     self.net, self.device, self.epochs,
        #     self.criterion, self.optimizer, self.lr_scheduler)
        embedding_xs = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)
        embedding_ys = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)
        embedding_both = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)

        projection_posterior = NonLinearMVN(
            MLP([128, 64], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True))

        output_model = nn.Sequential([
            ResBlock(
                MLP([128, 128], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True),
            ),
            ResBlock(
                MLP([128, 128], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True),
            ),
            nn.Dense(2)
        ])
        # output_model = MLP([128, 128, 2], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True)
        projection_outputs = NonLinearMVN(output_model)

        posterior_aggregator = MeanAggregator(projection_posterior)
        # posterior_aggregator = SequenceAggregator(projection_posterior)

        self.model = MixtureNeuralProcess(
            embedding_xs, embedding_ys, embedding_both,
            posterior_aggregator,
            projection_outputs
        )

        self.rng, self.key = jax.random.split(self.rng)
        self.params = self.model.init({'params': self.key, 'default': self.key}, xs[:, None], ys[:, None], xs[:3, None])

    # def _init_logger(self, algorithm_name, data_name,
    #                  net_name, num_epochs, random_seed):
    #     log_info = '%s-%s-%s-%d-%d-%s' % (
    #         algorithm_name, data_name, net_name, num_epochs, random_seed,
    #         time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
    #     self.log_dir = os.path.join('./runs', log_info)
    #     if not os.path.exists('./runs'): os.mkdir('./runs')
    #     if not os.path.exists(self.log_dir):
    #         os.mkdir(self.log_dir)
    #     else:
    #         print('The directory %s has already existed.' % (self.log_dir))
    #
    #     self.log_interval = 1
    #     self.logger = get_logger(os.path.join(self.log_dir, 'train.log'), log_info)

    @jax.jit
    def step(
            self,
            theta: flax.typing.VariableDict,
            opt_state: optax.OptState,
            random_key: flax.typing.PRNGKey
    ) -> tuple[flax.typing.VariableDict, optax.OptState, jax.Array]:
        # Implements a generic SGD Step

        # value, grad = jax.value_and_grad(posterior_loss_filtered, argnums=0)(theta, random_key)
        value, grad = jax.value_and_grad(self.posterior_loss, argnums=0)(theta, random_key)

        updates, opt_state = self.optimizer.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        return theta, opt_state, value

    def _train(self):
        best_acc = 0.0
        best, best_params = jnp.inf, self.params
        losses = list()
        for epoch in (pbar := tqdm.trange(self.epochs, desc='Optimizing params. ')):

        #for step, data in enumerate(loader):
            # inputs = data[0].to(self.device)
            # labels = data[1].to(self.device)
            # indices = data[2].to(self.device)
            #
            # self.optimizer.zero_grad()
            # outputs = net(inputs)
            # loss = self.loss_curriculum(  # curriculum part
            #     self.criterion, outputs, labels, indices)
            # loss.backward()
            # self.optimizer.step()
            #
            # train_loss += loss.item()
            # _, predicted = outputs.max(dim=1)
            # correct += predicted.eq(labels).sum().item()
            # total += labels.shape[0]
            self.rng, self.key = jax.random.split(self.rng)
            params_new, opt_state, loss = self.step(self.params, self.opt_state, self.key)

            losses.append(loss)

            if loss < best:
                best = loss
                best_params = params_new

            if jnp.isnan(loss):
                break
            else:
                self.params = params_new

            pbar.set_description(f'Optimizing params. Loss: {loss:.4f}')

            self.lr_scheduler.step()
            # self.logger.info(
            #     '[%3d]  Train data = %6d  Train Acc = %.4f  Loss = %.4f  Time = %.2f'
            #     % (epoch + 1, total, correct / total, train_loss / (step + 1), time.time() - t))

            # if (epoch + 1) % self.log_interval == 0:
            #     valid_acc = self._valid(self.valid_loader)
            #     if valid_acc > best_acc:
            #         best_acc = valid_acc
            #         torch.save(net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
            #     self.logger.info(
            #         '[%3d]  Valid data = %6d  Valid Acc = %.4f'
            #         % (epoch + 1, len(self.valid_loader.dataset), valid_acc))

    def posterior_loss(
            self,
            params: flax.typing.VariableDict,
            key: flax.typing.PRNGKey,
            sampling_fun: Callable[
                [flax.typing.PRNGKey],
                tuple[jax.Array, jax.Array]
            ] = data_sampler
    ):
        #TODO sample a batch from the loader
        # Sample datasets from p(X, Y, Z)
        key_data, key_model = jax.random.split(key)
        xs, ys = jax.vmap(data_sampler)(jax.random.split(key_data, num=batch_size))
        xs, ys = xs[..., None], ys[..., None]

        # Split into context- and target-points.
        X, x_test = jnp.split(xs, indices_or_sections=(num_context_samples,), axis=1)
        y, y_test = jnp.split(ys, indices_or_sections=(num_context_samples,), axis=1)

        # Compute ELBO over batch of datasets
        elbos = jax.vmap(partial(
            self.model.apply,
            params,
            beta=self.kl_penalty, k=self.num_posterior_mc,
            method=self.model.elbo
        ))(
            X, y, x_test, y_test, rngs={'default': jax.random.split(key_model, num=batch_size)}
        )

        return -elbos.mean()
    # def _valid(self, loader):
    #     total = 0
    #     correct = 0
    #
    #     self.net.eval()
    #     with torch.no_grad():
    #         for data in loader:
    #             inputs = data[0].to(self.device)
    #             labels = data[1].to(self.device)
    #
    #             outputs = self.net(inputs)
    #             _, predicted = jnp.max(outputs, dim=1)
    #             total += labels.shape[0]
    #             correct += predicted.eq(labels).sum().item()
    #     return correct / total
    #
    # def fit(self):
    #     set_random(self.random_seed)
    #     self._train()

    # def evaluate(self, net_dir=None):
    #     self._load_best_net(net_dir)
    #     valid_acc = self._valid(self.valid_loader)
    #     test_acc = self._valid(self.test_loader)
    #     self.logger.info('Best Valid Acc = %.4f and Final Test Acc = %.4f' % (valid_acc, test_acc))
    #     return test_acc
    #
    # def export(self, net_dir=None):
    #     self._load_best_net(net_dir)
    #     return self.net

    # def _load_best_net(self, net_dir):
    #     if net_dir is None: net_dir = self.log_dir
    #     net_file = os.path.join(net_dir, 'net.pkl')
    #     assert os.path.exists(net_file), 'Assert Error: the net file does not exist'
    #     self.net.load_state_dict(torch.load(net_file))