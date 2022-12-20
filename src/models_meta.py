import pdb
import collections
import torch

import learn2learn as l2l
from learn2learn.algorithms import MAML, maml_update
import traceback


def clone_module(module, memo=None):
    """
        [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

        **Description**

        Creates a copy of a module, whose parameters/buffers/submodules
        are created using PyTorch's torch.clone().

        This implies that the computational graph is kept, and you can compute
        the derivatives of the new modules' parameters w.r.t the original
        parameters.

        **Arguments**

        * **module** (Module) - Module to be cloned.

        **Return**

        * (Module) - The cloned module.

        **Example**

        ~~~python
        net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
        clone = clone_module(net)
        error = loss(clone(X), y)
        error.backward()  # Gradients are back-propagate all the way to net.
        ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned


    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)

    return clone

class NoCaptureCheckpoint(torch.autograd.Function):
    """
        To enable gradient checkpointing within this module, see this thread:
        https://github.com/pytorch/pytorch/issues/32005
    """
    @staticmethod
    def forward(ctx, fn, *args):
        ctx.save_for_backward(*args)
        ctx.fn = fn
        with torch.no_grad():
            outputs = fn(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grads):
        args = ctx.saved_tensors
        with torch.enable_grad():
            outputs = ctx.fn(*args)
        return (None,) + torch.autograd.grad(outputs, args, grads)

class MAMLS2SModel(l2l.algorithms.MAML):
    
    def adapt(self, loss, first_order=None, allow_unused=None, allow_nograd=None, gradient_checkpointing=False):
        """
            **Description**
            Takes a gradient step on the loss and updates the cloned parameters in place.
            **Arguments**
            * **loss** (Tensor) - Loss to minimize upon update.
            * **first_order** (bool, *optional*, default=None) - Whether to use first- or
                second-order updates. Defaults to self.first_order.
            * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
                of unused parameters. Defaults to self.allow_unused.
            * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
                parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        if gradient_checkpointing == False:
            gradients = self.compute_gradients(loss, first_order, allow_unused, allow_nograd)
        else:
            checkpoint_forward = NoCaptureCheckpoint.apply
            gradients = checkpoint_forward(self.compute_gradients, loss, first_order, allow_unused, allow_nograd)

        self.module = maml_update(self.module, self.lr, gradients)

    def compute_gradients(self, loss, first_order, allow_unused, allow_nograd):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = torch.autograd.grad(loss, diff_params, retain_graph=second_order, create_graph=second_order, allow_unused=allow_unused)
            gradients = []
            grad_counter = 0
            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = torch.autograd.grad(loss, self.module.parameters(), retain_graph=second_order, create_graph=second_order, allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

        return gradients

    def adapt_new(self, model, loss, use_amp, use_apex, deepspeed=False, first_order=None, allow_unused=None, allow_nograd=None, args=None, scaler=None):
        """
            **Description**
            Takes a gradient step on the loss and updates the cloned parameters in place.
            **Arguments**
            * **loss** (Tensor) - Loss to minimize upon update.
            * **first_order** (bool, *optional*, default=None) - Whether to use first- or
                second-order updates. Defaults to self.first_order.
            * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
                of unused parameters. Defaults to self.allow_unused.
            * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
                parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = torch.autograd.grad(loss,
                               diff_params,
                               retain_graph=second_order,
                               create_graph=second_order,
                               allow_unused=allow_unused)
            gradients = []
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                # gradients = torch.autograd.grad(loss,
                #                  self.module.parameters(),
                #                  retain_graph=second_order,
                #                  create_graph=second_order,
                #                  allow_unused=allow_unused)
                loss.backward(retain_graph=True, create_graph=True)
                gradients = [p.grad for p in model.__parameters if p.is_leaf()]

            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')
        # Update the module
        self.module = maml_update(self.module, self.lr, gradients)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
            **Description**

            Returns a `MAML`-wrapped copy of the module whose parameters and buffers
            are `torch.clone`d from the original module.

            This implies that back-propagating losses on the cloned module will
            populate the buffers of the original module.
            For more information, refer to learn2learn.clone_module().

            **Arguments**

            * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
                or second-order updates. Defaults to self.first_order.
            * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
            * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
                parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        cloned_model = MAMLS2SModel(clone_module(self.module), lr=self.lr, first_order=first_order, allow_unused=allow_unused, allow_nograd=allow_nograd)

        return cloned_model

    def sync_autograd(self, loss, parameters, use_amp, use_apex, deepspeed, second_order=False, args=None, scaler=None): # DDP and AMP compatible autograd
        # ---------- take backward step ------------
        loss_backward_done = False
        if use_amp:
            # scaler.scale(loss).backward(create_graph=second_order)
            loss = scaler.scale(loss)
            loss_backward_done = True
        # elif use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward(retain_graph=second_order, create_graph=second_order)
        #         loss_backward_done = True
        # elif deepspeed:
        #     # loss gets scaled under gradient_accumulation_steps in deepspeed
        #     loss_backward_done = True
        #     loss = self.deepspeed.backward(loss)
        # else:

        gradients = torch.autograd.grad(loss, parameters, retain_graph=second_order, create_graph=second_order, allow_unused=self.allow_unused)
        # # ---------- Gradient clipping ----------
        # if args.max_grad_norm_meta is not None and args.max_grad_norm_meta > 0 and not deepspeed: # deepspeed does its own clipping
        #     if use_amp:# AMP: gradients need unscaling
        #         scaler.unscale_(self.inner_optimizer)
        #     nn.utils.clip_grad_norm_(amp.master_params(self.inner_optimizer) if use_apex else parameters, args.max_grad_norm_meta,)
        # ------------- copy gradients ------------
        # probably do not need to clone these
        # gradients = [p.grad.clone() for p in parameters]
        # gradients = [p.grad for p in parameters]

        # if loss_backward_done == True:
        #     gradients = [p.grad.clone() for p in parameters]

        return gradients



#!/usr/bin/env python3

import torch as th
from torch import nn
from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, clone_parameters


def meta_sgd_update_old(model, lrs=None, grads=None):
    """
        **Description**

        Performs a MetaSGD update on model using grads and lrs.
        The function re-routes the Python object, thus avoiding in-place
        operations.

        NOTE: The model itself is updated in-place (no deepcopy), but the
            parameters' tensors are not.

        **Arguments**

        * **model** (Module) - The model to update.
        * **lrs** (list) - The meta-learned learning rates used to update the model.
        * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
            of the model. If None, will use the gradients in .grad attributes.

        **Example**
        ~~~python
        meta = l2l.algorithms.MetaSGD(Model(), lr=1.0)
        lrs = [th.ones_like(p) for p in meta.model.parameters()]
        model = meta.clone() # The next two lines essentially implement model.adapt(loss)
        grads = autograd.grad(loss, model.parameters(), create_graph=True)
        meta_sgd_update(model, lrs=lrs, grads)
        ~~~
    """

    if grads is not None and lrs is not None:
        for p, lr, g in zip(model.parameters(), lrs, grads):
            p.grad = g
            p._lr = lr

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - p._lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None and buff._lr is not None:
            model._buffers[buffer_key] = buff - buff._lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = meta_sgd_update(model._modules[module_key])
    return model


def meta_sgd_update(model, lrs=None, grads=None):
    """
        **Description**

        Performs a MetaSGD update on model using grads and lrs.
        The function re-routes the Python object, thus avoiding in-place
        operations.

        NOTE: The model itself is updated in-place (no deepcopy), but the
            parameters' tensors are not.

        **Arguments**

        * **model** (Module) - The model to update.
        * **lrs** (list) - The meta-learned learning rates used to update the model.
        * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
            of the model. If None, will use the gradients in .grad attributes.

        **Example**
        ~~~python
        meta = l2l.algorithms.MetaSGD(Model(), lr=1.0)
        lrs = [th.ones_like(p) for p in meta.model.parameters()]
        model = meta.clone() # The next two lines essentially implement model.adapt(loss)
        grads = autograd.grad(loss, model.parameters(), create_graph=True)
        meta_sgd_update(model, lrs=lrs, grads)
        ~~~
    """
    if lrs is not None:
        for i, (n, p) in enumerate(model.named_parameters()):
            lr = getattr(lrs, n.replace(".", "#"))
            p.grad = grads[i]
            p._lr = lr
            # # if i==0:
            # #     print(p._lr)
            # try:
            #     assert(p.shape==lr.shape)
            #     if grads[i] is not None:
            #         assert(p.shape==grads[i].shape)
            # except:
            #     print(p.shape, lr.shape, grads[i].shape)
            #     pdb.set_trace()

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            try:
                model._parameters[param_key] = p - p._lr * p.grad
            except:
                pdb.set_trace()
    # Second, handle the buffers if necessary
    #  This is not really used here
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None and buff._lr is not None:
            model._buffers[buffer_key] = buff - buff._lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = meta_sgd_update(model._modules[module_key])
    return model

class PerParamLRS(torch.nn.Module):
    def __init__(self, model, lrs, lr=1.0, per_param_per_layer=False):
        super().__init__()
        self.lr = lr
        self.per_param_per_layer = per_param_per_layer
        if lrs is None:
            for n, p in model.named_parameters():
                # if per_param_per_layer:
                #     #  create a learning rate per param per layer of main model
                #     setattr(self, n.replace(".", "#"), nn.Parameter(th.ones_like(p) * lr))
                # else:
                #     #  create a learning rate per layer of main model
                #     setattr(self, n.replace(".", "#"), nn.Parameter(torch.FloatTensor(lr)))
                self.add_parameter(p, n)
    
    def add_parameter(self, p, name, device=None):
        updated_name = name.replace(".", "#")
        if self.per_param_per_layer:
            #  create a learning rate per-param per layer of main model
            setattr(self, updated_name, nn.Parameter(th.ones_like(p) * self.lr))
        else:
            #  create a learning rate per layer of main model
            setattr(self, updated_name, nn.Parameter(torch.FloatTensor([self.lr])))
              
class MetaSGD(BaseLearner):
    """
            [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/meta_sgd.py)

            **Description**

            High-level implementation of *Meta-SGD*.

            This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt`
            methods.
            It behaves similarly to `MAML`, but in addition a set of per-parameters learning rates
            are learned for fast-adaptation.

            **Arguments**

            * **model** (Module) - Module to be wrapped.
            * **lr** (float) - Initialization value of the per-parameter fast adaptation learning rates.
            * **first_order** (bool, *optional*, default=False) - Whether to use the first-order version.
            * **lrs** (list of Parameters, *optional*, default=None) - If not None, overrides `lr`, and uses the list
                as learning rates for fast-adaptation.

            **References**

            1. Li et al. 2017. “Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.” arXiv.

            **Example**

            ~~~python
            linear = l2l.algorithms.MetaSGD(nn.Linear(20, 10), lr=0.01)
            clone = linear.clone()
            error = loss(clone(X), y)
            clone.adapt(error)
            error = loss(clone(X), y)
            error.backward()
            ~~~
    """

    def __init__(self, model, lr=1.0, first_order=False, lrs=None, per_param_per_layer=False):
        super(MetaSGD, self).__init__()
        if lrs is None:
            lrs = PerParamLRS(model, lrs, lr=lr, per_param_per_layer=per_param_per_layer)
        self.lrs = lrs 
        self.module = model
        self.first_order = first_order

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(self):
        """
            **Descritpion**

            Akin to `MAML.clone()` but for MetaSGD: it includes a set of learnable fast-adaptation
            learning rates.
        """
        return MetaSGD(clone_module(self.module), lrs=clone_module(self.lrs), first_order=self.first_order)

    def adapt(self, loss, first_order=None):
        """
            **Descritpion**

            Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
            per-parameter learning rates.

            elif self.lr_type == 'per_parameter': # As proposed in "Meta-SGD".
                self.lr = nn.ParameterList([])
                hypo_parameters = hypo_module.parameters()
                for param in hypo_parameters:
                    self.lr.append(nn.Parameter(torch.ones(param.size()) * init_lr))
            elif self.lr_type == 'per_parameter_per_step':
                self.lr = nn.ModuleList([])
                for name, param in hypo_module.meta_named_parameters():
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
            for _ in range(num_meta_steps)]))

        """
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        gradients = grad(loss,
                         self.module.parameters(),
                         retain_graph=second_order,
                         create_graph=second_order, allow_unused=True)

        # update lrs with additional parameters if any
        # main_model_params = [n for n, p in self.module.named_parameters()]
        # get the original module names for lrs_params
        lrs_params = [n.replace("#", ".") for n, p in self.lrs.named_parameters()]
        for name, p in self.module.named_parameters():
            if name not in lrs_params:
                self.lrs.add_parameter(p, name, device=p.device)
                self.lrs.to(device=p.device)

        self.module = meta_sgd_update(self.module, self.lrs, gradients)

if __name__ == '__main__':
    linear = nn.Sequential(nn.Linear(10, 2), nn.Linear(5, 5))
    msgd = MetaSGD(linear, lr=0.001)
    learner = msgd.new()
