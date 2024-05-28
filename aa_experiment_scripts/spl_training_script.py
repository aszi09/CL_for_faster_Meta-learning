



from typing import Callable, Sequence, Any
from functools import partial
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from aa_train_utils.model_utils import create_model, save_model_params, load_model_params
from aa_train_utils.dataset_generation import joint, uniform, f6, f5, RegressionDataset , generate_noisy_split_trainingdata
from aa_train_utils.spl_curriculum import SPL_curriculum 

import jax
import jax.numpy as jnp
import jax.tree_util
from jax.scipy.stats.norm import logpdf
import pickle
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch

import numpy as np
import pickle
import flax
import flax.linen as nn

import optax
import jaxopt
import netket as nk

import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from functions import Fourier, Mixture, Slope, Polynomial, WhiteNoise, Shift
from networks import MixtureNeuralProcess, MLP, MeanAggregator, SequenceAggregator, NonLinearMVN, ResBlock
#from dataloader import MixtureDataset

from jax.tree_util import tree_map
from torch.utils import data




def cross_entropy_error(model, params, x_context, y_context, x_target, y_target , rng , k):
    full_x = jnp.concatenate([x_context, x_target])
    y_means, y_stds = model.apply(params, x_context, y_context, full_x,k=k, rngs={'default': rng})

    full_y = jnp.concatenate([y_context, y_target])

    # Lets compute the log likelihood of the target points given the means and stds

    #print(full_y.shape, y_means.shape, y_stds.shape, "printing the shapes, they should be just an array of values")
    log_pdf = logpdf(full_y, jnp.squeeze(y_means),jnp.squeeze(y_stds)) 
    return -jnp.mean(log_pdf)



def RMSE_means(model, params, x_context, y_context, x_target, y_target, rng, k):
    
    full_x = jnp.concatenate([x_context, x_target])
    y_means, y_stds = model.apply(params, x_context, y_context, full_x,k=k, rngs={'default': rng}) 
    
    full_y = jnp.concatenate([y_context, y_target])
    
    return jnp.sqrt(jnp.mean((y_means - full_y)**2))


def STD_residuals(model, params, x_context, y_context, x_target, y_target, rng, k):
    
    full_x = jnp.concatenate([x_context, x_target])
    y_means, y_stds = model.apply(params, x_context, y_context, full_x,k=k, rngs={'default': rng}) 
    
    full_y = jnp.concatenate([y_context, y_target])

    return abs(full_y - y_means) / y_stds 



def train_spl_curriculum(dataset_key_int,dataloader_key_int, dataset_size, training_step_number, eval_dataset_size, eval_intervals, sampler_ratios, chunk_size, save_path ,  model_name, start_rate, growth_epochs):
    
    """ Training function for the SPL curriculum based Neural Process model training"""



    # Lets define the training functions here and not in their own files, because I couldnt make them modular enough.
    # (The posterior loss was relying on the global variable model, I tried creating a partial with the params not included to have the scan carry over a new param based partial to the step function but it wasnt working, this works for now)

    def posterior_loss(
        params: flax.typing.VariableDict,
        batch,
        key: flax.typing.PRNGKey,
    ):
        key_data, key_model = jax.random.split(key)
        


        X = batch[0]
        y = batch[1]
        x_test = batch[2]
        y_test = batch[3]
        # Compute ELBO over batch of datasets
        elbos = jax.vmap(
        partial(
                model.apply,
                params,  
                beta=kl_penalty,
                k=num_posterior_mc,
                method=model.elbo
        ) 
        )(
            X, y, x_test, y_test, rngs={'default': jax.random.split(key_model, X.shape[0])}
        )
        
        return -elbos.mean()

    @jax.jit
    def step(
        theta: flax.typing.VariableDict, 
        opt_state: optax.OptState,
        current_batch,
        random_key: flax.typing.PRNGKey,
    ) -> tuple[flax.typing.VariableDict, optax.OptState, jax.Array]:
        # Implements a generic SGD Step
        
        # value, grad = jax.value_and_grad(posterior_loss_filtered, argnums=0)(theta, random_key)
        value, grad = jax.value_and_grad(posterior_loss, argnums=0)(theta, current_batch, random_key )
        
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        
        return theta, opt_state, value


    def body_batch(carry, batch):
        params, opt_state, key = carry
        key_carry, key_step = jax.random.split(key)

        X, x_test = jnp.split(batch[0], indices_or_sections=(num_context_samples, ), axis=1)
        y, y_test = jnp.split(batch[1], indices_or_sections=(num_context_samples, ), axis=1)
        params, opt_state, value = step(params, opt_state, (X,y, x_test,y_test ), key_step )

        return (params, opt_state, key_carry ), value

    jax.jit
    def scan_train(params, opt_state, key,  batches):
        
        last, out = jax.lax.scan(body_batch, (params, opt_state, key ), batches)

        params, opt_state, _ = last
        
        return params, opt_state, out

    torch.manual_seed(dataloader_key_int) # Setting the seed for the dataloader
    os.makedirs(save_path, exist_ok=True)
    num_context_samples = 64
    num_target_samples = 32
    batch_size = 128
    kl_penalty = 1e-4
    num_posterior_mc = 1


    # First lets create the dataset, 
    # Lets hardcode it for now, and then we can make it more flexible later on
    
    sampler_noise = partial(
        joint, 
        WhiteNoise(f6, 0.1), 
        partial(uniform, n=num_target_samples + num_context_samples, bounds=(-1, 1))
    )

    sampler_clean = partial(
        joint, 
        f6, 
        partial(uniform, n=num_target_samples + num_context_samples, bounds=(-1, 1))
    )

    out_task_sampler = partial(
        joint, 
        f5, 
        partial(uniform, n=num_target_samples + num_context_samples, bounds=(-1, 1))
    )
    samplers = [sampler_noise, sampler_clean]

    dataset_key = jax.random.PRNGKey(dataset_key_int)
    dataset = RegressionDataset(generate_noisy_split_trainingdata(samplers, sampler_ratios, dataset_size, chunk_size , dataset_key))

    # Lets setup the SPL curriculum

    rng , curricula_key = jax.random.split(dataset_key)
    spl_curricula = SPL_curriculum(start_rate, growth_epochs , dataset, batch_size, curricula_key)



    # Lets initalize the model we are going to train

    rng, key = jax.random.split(rng)

    model , params = create_model(key)
    optimizer = optax.chain(
        optax.clip(.1),
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=1e-6),
    )
    opt_state = optimizer.init(params)

    best, best_params = jnp.inf, params
    losses = list()
    in_task_errors = {'ece':[], 'rmse':[], 'std_residuals':[]} # We will log the in task errors for the model
    out_task_errors = {'ece':[], 'rmse':[], 'std_residuals':[]} # We will log the out of task errors for the model
    training_steps = 0

    for i in (pbar := tqdm.trange(10 ,desc='Optimizing params. ')):
        
        rng, key = jax.random.split(rng)
        _ , eval_epoch_key = jax.random.split(rng)
        model_partial_loss_function = partial(model.apply, params, beta=kl_penalty, k=num_posterior_mc, method=model.elbo) 
        


        batches = jnp.asarray( jax.tree_util.tree_map(lambda tensor : tensor.numpy(), [batch for batch in spl_curricula.data_curriculum(model_partial_loss_function, i, num_context_samples)]))
        batches_until_eval = eval_intervals - (training_steps % eval_intervals)
        batches_until_end = training_step_number - training_steps
        if batches_until_end < len(batches):
            batches = batches[:batches_until_end]

        #print("batches_until_eval", batches_until_eval, "batches_until_end", batches_until_end, "len(batches)", len(batches), "training_steps", training_steps )

        
        if batches_until_eval < len(batches):
            # then get the slice to make up the eval_intervals
            
            trained_steps_within_eval = 0
             
            batch_slice_pre_eval = eval_intervals - ( training_steps % eval_intervals )
            batch_slice = batch_slice_pre_eval 
            loss_array_eval = []
            params_new = params
            for i in range(0,1+((len(batches)-batch_slice_pre_eval) // eval_intervals)):
                

                #print("current eval loop number", i , "currently trained steps within eval", trained_steps_within_eval , "current batch slice", (trained_steps_within_eval, (trained_steps_within_eval+batch_slice)) ) 
                params_new, opt_state, loss_arr = scan_train(params_new, opt_state, key,batches[trained_steps_within_eval:(trained_steps_within_eval+batch_slice)])

                loss_array_eval.extend(loss_arr)  # dont lose the loss values upon next batch training
                trained_steps_within_eval += batch_slice
                batch_slice = eval_intervals

                eval_epoch_key, eval_inkey_data, eval_outkey_data, eval_model_key = jax.random.split(eval_epoch_key, 4)
                intask_x_eval, intask_y_eval = jax.vmap(sampler_clean)(jax.random.split(eval_inkey_data, eval_dataset_size)) 
                intask_x_eval, intask_y_eval = intask_x_eval[..., None], intask_y_eval[..., None]

                #lets split them into the context and target sets
                x_contexts, x_targets = jnp.split(intask_x_eval, indices_or_sections=(num_context_samples, ), axis=1)
                y_contexts, y_targets = jnp.split(intask_y_eval, indices_or_sections=(num_context_samples, ), axis=1)

                ece_errors = jax.vmap(partial(cross_entropy_error, model, params_new, k=num_posterior_mc), in_axes=(0,0,0,0,0))(x_contexts, y_contexts, x_targets, y_targets, jax.random.split(eval_model_key, eval_dataset_size))
                rmse_errors= jax.vmap(partial(RMSE_means, model, params_new, k=num_posterior_mc), in_axes=(0,0,0,0,0))(x_contexts, y_contexts, x_targets, y_targets, jax.random.split(eval_model_key, eval_dataset_size))
                std_residuals= jax.vmap(partial(STD_residuals, model, params_new, k=num_posterior_mc), in_axes=(0,0,0,0,0))(x_contexts, y_contexts, x_targets, y_targets, jax.random.split(eval_model_key, eval_dataset_size))

                in_task_errors['ece'].append(ece_errors.mean())
                in_task_errors['rmse'].append(rmse_errors.mean())
                in_task_errors['std_residuals'].append(std_residuals.mean())

                # Now lets do the out of task evaluation (f for now like the original notebook)
                outtask_x_eval, outtask_y_eval = jax.vmap(out_task_sampler)(jax.random.split(eval_outkey_data, eval_dataset_size))
                outtask_x_eval, outtask_y_eval = outtask_x_eval[..., None], outtask_y_eval[..., None]

                #lets split them into the context and target sets
                x_contexts, x_targets = jnp.split(outtask_x_eval, indices_or_sections=(num_context_samples, ), axis=1)
                y_contexts, y_targets = jnp.split(outtask_y_eval, indices_or_sections=(num_context_samples, ), axis=1)

                ece_errors = jax.vmap(partial(cross_entropy_error, model, params_new, k=num_posterior_mc), in_axes=(0,0,0,0,0))(x_contexts, y_contexts, x_targets, y_targets, jax.random.split(eval_model_key, eval_dataset_size))
                rmse_errors= jax.vmap(partial(RMSE_means, model, params_new, k=num_posterior_mc), in_axes=(0,0,0,0,0))(x_contexts, y_contexts, x_targets, y_targets, jax.random.split(eval_model_key, eval_dataset_size))
                std_residuals= jax.vmap(partial(STD_residuals, model, params_new, k=num_posterior_mc), in_axes=(0,0,0,0,0))(x_contexts, y_contexts, x_targets, y_targets, jax.random.split(eval_model_key, eval_dataset_size))

                out_task_errors['ece'].append(ece_errors.mean())
                out_task_errors['rmse'].append(rmse_errors.mean())
                out_task_errors['std_residuals'].append(std_residuals.mean())


                

            # Now we can train the rest of the batches
            
            # with trained_steps_within_eval start slicing, only train if len(batches) - trained_steps_within_eval > 0
            if len(batches) - trained_steps_within_eval > 0: 
                #print("training on the rest of the remaining batches after eval", len(batches)-trained_steps_within_eval)
                params_new , opt_state, loss_arr = scan_train(params_new, opt_state, key,batches[trained_steps_within_eval:])
                loss_array_eval.extend(loss_arr)
            #else: 
                #print("Eval period was the last period, no more training, eval intervals fit perfectly within this batch.")

            
            loss_arr = jnp.asarray(loss_array_eval)
        else: 
            params_new, opt_state, loss_arr = scan_train(params, opt_state, key,batches)
        
        # Update the training steps
        # Since this variable is only used inside the function and never later , it doesnt matter for the training_step_number restriction if it overcounts.  
        # Although it would so pay attention if implementation changes.
        #print(training_steps, len(batches))
        training_steps+= len(batches)

    

        

        losses.extend(loss_arr)

        if loss_arr.min() < best:
            best = loss_arr.min()
            best_params = params_new
        
        if jnp.isnan(loss_arr).any():
            break
        else:
            params = params_new
        
        pbar.set_description(f'Optimizing params. Loss: {loss_arr.min():.4f}')

        if(training_steps >= training_step_number):
            break
    # Lets save what we need to save for the model and training. 

    ### After training we should save  
        # the model parameters with a name that we know how it was trained
        # the losses and other relevant information accrued during training
        # the curriculum weight log for the dataset

    # Saving the model params
    # We could also save opt_state here for later training
    # Also after trying out whether the training would continue saving and loading the params back in I saw change in printed loss. Not sure why that is the case,
    # Even if I restore the opt_state as well. Regardless, the model continued training so saving the params is enough to use the model for evaluation later on. 
    
    print("printing losses length", len(losses))
    save_model_params(best_params,save_path, model_name) 
    
    with open(os.path.join(save_path, model_name + '_curricula_logs.pkl'), 'wb') as f:
        if(len(spl_curricula.weight_log)>0):
            pickle.dump({"curricula_weights": spl_curricula.weight_log , "curricula_losses": spl_curricula.epoch_losses_log}, f)
    
    with open(os.path.join(save_path, model_name + '_training_metrics.pkl'), 'wb') as f:
        pickle.dump({"training_loss" : losses, "training_intask_errors": in_task_errors, "training_outtask_errors": out_task_errors }, f)




train_spl_curriculum(0,0, 128*100, 300, 100, 128, [0.3,0.7], 128, "./spl_training_data/", "spl_model_0", 0.1, 5)