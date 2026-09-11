"""Shared optimization/logging plumbing used by the ctrex sample-model classes.

Defines `OptimModule`, a `nn.Module` mixin giving subclasses per-parameter learning rates
and a decorator-based mechanism (`register_loss`/`register_metric`) for registering loss
and metric functions that get automatically collected across a whole module tree (e.g. by
`CTReconstruction`). Also provides small TensorBoard logging helpers (`TBLogger`,
`SummaryWriter`) used to work around a PyTorch Lightning/TensorBoard hparams logging issue.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, List
import inspect

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch.utils.tensorboard as tb
from torch.utils.tensorboard.summary import hparams


def call_with_optional_arg(fn, arg):
    """Call `fn()` if it takes no arguments, otherwise `fn(arg)`. Lets registered loss/metric
    functions optionally accept a reference/ground-truth module without every one having to
    declare an unused parameter."""
    sig = inspect.signature(fn)
    if len(sig.parameters) == 0:
        return fn()
    return fn(arg)

@dataclass
class LossSpec:
    """Wraps one `@register_loss`-decorated method with the module that owns its weight
    attribute (`<name>_weight`), so `value()` can look up the current weight and apply it."""
    name: str
    compute: Callable
    module: object

    @property
    def weight(self):
        """Resolve the loss weight from the module."""
        return getattr(self.module, self.weight_name, None)

    @property
    def weight_name(self):
        return f"{self.name}_weight"


    def value(self, ref_module=None) -> dict:
        """ Calling the registered loss method via a LossSpec gives the weighted value,
        but calling the registered loss method itself will give the unweighted raw value """

        if self.weight is None or self.weight == 0:
            return {self.name: None}

        val = call_with_optional_arg(self.compute, ref_module)
        return {self.name: self.weight * val}


@dataclass
class MetricSpec:
    """Wraps one `@register_metric`-decorated method. Unlike `LossSpec`, metrics are always
    computed (no weight/gating) and are meant for logging/inspection rather than backprop."""
    name: str
    compute: Callable

    def value(self, gt_module=None) -> dict:
        """Compute the metric. If the underlying function returns a dict, its keys are
        namespaced under this metric's name (`<name>/<key>`); a scalar result is wrapped
        as `{name: result}`."""
        result = call_with_optional_arg(self.compute, gt_module)

        if isinstance(result, dict):
            # Namespace keys under this metric name
            return {
                f"{self.name}/{k}": v
                for k, v in result.items()
            }

        # Scalar → wrap into dict
        return {self.name: result}


def register_loss(fn):
    """Decorator marking a method as a loss contribution. `OptimModule.__init_subclass__`
    collects all such methods into `cls._loss_fns` so they're picked up automatically by
    `collect_losses` on any module in the tree, weighted by a `<method_name>_weight`
    attribute on the owning instance."""
    fn._is_loss = True
    return fn

def register_metric(fn):
    """Decorator marking a method as a metric to log (see `register_loss`, but metrics are
    always computed and never weighted)."""
    fn._is_metric = True
    return fn


class OptimModule(nn.Module):
    """Mixin base class (subclass of `nn.Module`) providing the optimization/logging
    plumbing shared by the ctrex sample-model classes (e.g. `CTSimulation`, shape/track
    models, projection effects).

    Two independent features are bundled here:

    - Loss/metric registration: methods decorated with `@register_loss`/`@register_metric`
      anywhere in a module tree are discovered via `__init_subclass__` and can be collected
      in one call with `collect_losses`/`collect_metrics`, which walk `self.modules()` and
      aggregate results from every submodule that defines `loss_specs`/`metric_specs`.
    - Per-parameter learning rates: `learning_rate`/`learning_rates` let a module (or its
      submodules) specify a custom learning rate per named parameter; `params_to_optimize`
      turns the whole tree into optimizer param groups honoring those rates.

    Also provides state-dict helpers (`small_state_dict`, `save_state_dict`,
    `load_small_state_dict`) for saving/restoring only the small, "interesting" parameters
    and buffers (e.g. for debugging/inspection), skipping large tensors.
    """
    _loss_fns = []
    _metric_fns = []

    def __init_subclass__(cls):
        """Collect every `@register_loss`/`@register_metric`-decorated method defined on
        `cls` into `cls._loss_fns`/`cls._metric_fns` (extending whatever the parent class
        already collected), so `loss_specs`/`metric_specs` don't need to re-scan at runtime."""
        super().__init_subclass__()

        cls._loss_fns = list(getattr(cls, "_loss_fns", []))
        cls._metric_fns = list(getattr(cls, "_metric_fns", []))

        for attr in cls.__dict__.values():
            if callable(attr):
                if getattr(attr, "_is_loss", False):
                    cls._loss_fns.append(attr)
                if getattr(attr, "_is_metric", False):
                    cls._metric_fns.append(attr)

    def __init__(self, learning_rate = None):
        """
        Args:
            learning_rate: optional initial learning rate(s), forwarded to the
                `learning_rate` setter (a single value applied to every parameter, or a
                dict/list matching `named_parameters()`).
        """
        super(OptimModule, self).__init__()
        self._learning_rates = {}

        if learning_rate is not None:
            self.learning_rate = learning_rate

    def loss_specs(self):
        """Wrap this instance's registered loss methods (`_loss_fns`) as bound `LossSpec`s."""
        return [LossSpec(name=fn.__name__, compute=fn.__get__(self), module=self)
            for fn in self._loss_fns]

    def metric_specs(self):
        """Wrap this instance's registered metric methods (`_metric_fns`) as bound `MetricSpec`s."""
        return [MetricSpec(name=fn.__name__, compute=fn.__get__(self))
            for fn in self._metric_fns]

    def collect_specs(self, spec_attr: str) -> List[LossSpec | MetricSpec]:
        """Gather `spec_attr` (`'loss_specs'` or `'metric_specs'`) from every submodule in
        this module's tree (including self) that defines it, flattened into one list."""
        return [
            spec
            for module in self.modules()
            if hasattr(module, spec_attr)
            for spec in getattr(module, spec_attr)()
        ]

    def collect_spec_vals(self, spec_attr = 'metric_specs', other_module=None, prefix = '',
                          skip_none = True):
        """Compute and collect all loss or metric values across the module tree into one
        flat, prefixed dict.

        Args:
            spec_attr: which kind of spec to collect, `'loss_specs'` or `'metric_specs'`.
            other_module: if given, a parallel module tree (e.g. ground truth) walked
                alongside this one via `named_modules()`, passed as the reference/comparison
                argument to each spec's `value()` (only modules that themselves define
                `spec_attr` contribute; the two trees are assumed to line up structurally).
                If None, specs are evaluated with no reference argument.
            prefix: string prepended to every result key.
            skip_none: drop entries whose value is None (e.g. an unweighted/disabled loss).

        Returns:
            dict mapping prefixed spec name to its computed value.
        """
        spec_vals = {}
        def append_prefix(some_dict):
            some_dict = {prefix + key: val for key, val in some_dict.items()}
            if skip_none:
                some_dict = {key: val for key, val in some_dict.items() if val is not None}
            return some_dict


        if other_module is None:
            for spec in self.collect_specs(spec_attr):
                spec_val = spec.value(None)
                spec_vals.update(append_prefix(spec_val))

            return spec_vals

        # Specs that compare against a GT module
        for (_, mod_r), (_, mod_g) in zip(
                self.named_modules(), other_module.named_modules()
        ):
            if not hasattr(mod_r, spec_attr):
                continue

            for spec in getattr(mod_r, spec_attr)():
                spec_val = spec.value(mod_g)
                spec_vals.update(append_prefix(spec_val))

        return spec_vals

    def collect_losses(self, other_module=None, prefix = ''):
        """Collect all registered loss values across this module tree (see `collect_spec_vals`)."""
        return self.collect_spec_vals('loss_specs', other_module, prefix)

    def collect_metrics(self, other_module=None, prefix = ''):
        """Collect all registered metric values across this module tree (see `collect_spec_vals`)."""
        return self.collect_spec_vals('metric_specs', other_module, prefix)

    def collect_loss_weight_hparams(self) -> dict:
        """Collect the current weight of every registered loss across the module tree
        (skipping unset weights), keyed by `<loss_name>_weight`, for logging as hyperparameters."""
        hparams_dict = {spec.weight_name: spec.weight
                        for spec in self.collect_specs('loss_specs')
                        if spec.weight is not None}
        return hparams_dict


    @register_loss
    def default_loss(self):
        """Always-present zero loss, so `collect_losses` never returns an empty dict for a
        module with no other registered losses (e.g. `sum(loss_dict.values())` stays valid)."""
        return torch.tensor(0)

    @property
    def learning_rates(self):
        return self._learning_rates

    @learning_rates.setter
    def learning_rates(self, learning_rates):
        """Set per-parameter learning rates for this module's named parameters.

        Args:
            learning_rates: either a dict mapping parameter name to learning rate (only
                matching names are applied), a list/tuple assigning rates positionally
                (matched by zip against `named_parameters()`, in order), or a single value
                applied to every parameter.
        """
        names = [name for name, _ in self.named_parameters(recurse = True)]
        if isinstance(learning_rates, dict):
            parsed_learning_rates = {name: learning_rates[name] for name in names if name in learning_rates}
        elif isinstance(learning_rates, (tuple, list)):
            parsed_learning_rates = {name: lr for name, lr in zip(names, learning_rates)}
        else:  # single value
            parsed_learning_rates = {name: learning_rates for name in names}
        parsed_learning_rates = {name: lr for name, lr in parsed_learning_rates.items()}
        self._learning_rates.update(parsed_learning_rates)
        pass

    @property
    def learning_rate(self):
        """Alias for `learning_rates` (singular name reads better when setting one value)."""
        return self._learning_rates

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.learning_rates = learning_rate

    def set_requires_grad(self, requires_grad=False):
        """Enable/disable gradient tracking for every parameter in this module (recursively)."""
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    def params_to_optimize(self, default_lr = 1e-3):
        """Build optimizer param groups for every trainable parameter in this module's tree,
        one group per parameter so each can carry its own learning rate.

        Args:
            default_lr: learning rate used for parameters that have no per-parameter rate
                set via `learning_rates` (or whose owning module doesn't define `learning_rates`
                at all). A parameter with an explicit rate of 0 (or None resolved to 0) is
                skipped entirely, i.e. excluded from optimization.

        Returns:
            list of dicts, each `{"params": [param], "lr": lr, "module_name": ..., "name": ...}`,
            suitable for passing straight to a torch optimizer.
        """
        param_groups = []
        for module_name, module in self.named_modules():
            # only look at parameters owned by this module
            for name, param in module.named_parameters(recurse = False):
                if not param.requires_grad:
                    continue
                if hasattr(module, 'learning_rates'):
                    lr = module.learning_rates.get(name, default_lr)
                else:
                    lr = default_lr
                lr = lr if lr is not None else default_lr

                if lr is None or lr == 0:
                    continue

                param_groups.append({"params": [param], "lr": lr, "module_name": module_name,"name":name})

        return param_groups

    def print_params_to_optimize(self, **kwargs):
        """Print one line per optimized parameter (module name, parameter name, learning
        rate) - a manual debugging helper.

        NOTE: not called anywhere in src/ or scripts/; appears intended for interactive/
        debugging use only.
        """
        params_to_optimize = self.params_to_optimize(**kwargs)
        print("\n".join([f"{'.'.join(params['module_name'].split('.')[-2:]):<35}"  # module and potential parent module
                         f"\t{params['name']:<20}"  # parameter name
                         f"\tlr {params['lr']}"  # learning rate
                         for params in params_to_optimize]))

    def hparams(self, key = "lr"):
        """Build a flat dict of hyperparameters for logging/inspection: one `lr_<name>` entry
        per optimized parameter with a nonzero rate, plus (if present on this module's
        `projectors`) particle count and static-matrix ROI shape/voxel-scale info.

        Note: the `self.projectors` lookups assume attributes provided by specific
        `OptimModule` subclasses (e.g. `CTSimulation`), not by `OptimModule` itself.
        """
        params_to_optimize = self.params_to_optimize()
        hparam_dict = OrderedDict([(f'{key}_{param_group["name"]}', param_group[key])
                                   for param_group in params_to_optimize if param_group["lr"] > 0])
        if hasattr(self.projectors, 'particles'):
            hparam_dict.update(num_particles = self.projectors.particles.sample_component.num_particles)
        if hasattr(self.projectors, 'static_matrix'):
            hparam_dict.update(voxel_scale = self.projectors.static_matrix.sample_component.shape_model.voxel_scale,
                               roi_shape_z = int(self.projectors.static_matrix.sample_component.shape_model.shape[0]),
                               roi_shape_y = int(self.projectors.static_matrix.sample_component.shape_model.shape[1]),
                               roi_shape_x = int(self.projectors.static_matrix.sample_component.shape_model.shape[2]),
                               )
        return hparam_dict

    def small_state_dict(self, numel = 1000, ignore_words = ()):
        """Collect this module's parameters and buffers, keeping only the "small" ones (fewer
        than `numel` elements) whose name doesn't contain any of `ignore_words` - meant to
        capture per-particle/track-style state cheaply while excluding large tensors (e.g.
        full volumes) and duplicated/uninteresting entries.

        Args:
            numel: exclude any tensor with this many elements or more.
            ignore_words: exclude any parameter/buffer whose name contains one of these
                substrings.

        Returns:
            dict mapping name (with `.` replaced by `/`) to tensor.
        """
        # parameters with requires_grad, not tensors
        state_dict = dict(list(self.named_parameters()) + list(self.named_buffers()))  # parameters with requires_grad, not tensors
        state_dict = {name.replace('.', '/'): param
                      for name, param in state_dict.items()
                      if (param.numel() < numel
                      and not any(word in name for word in ignore_words))}  # duplicate trajectories
        return state_dict

    def save_state_dict(self, path, state_dict = None, **kwargs):
        """Save `state_dict` (or, if not given, `small_state_dict(**kwargs)`) to `path` via
        `torch.save`.

        NOTE: not called anywhere in src/ or scripts/; appears intended for interactive/
        debugging use only.
        """
        if state_dict is None:
            state_dict = self.small_state_dict(**kwargs)
        torch.save(state_dict, path)
        return state_dict

    def load_small_state_dict(self, state_dict_path):
        """Load a state dict previously saved via `save_state_dict`/`small_state_dict` from
        `state_dict_path` and apply it with `load_state_dict`.

        NOTE: not called anywhere in src/ or scripts/; appears intended for interactive/
        debugging use only.
        """
        small_state_dict = torch.load(state_dict_path, weights_only=True)
        return self.load_state_dict(small_state_dict)


def tensor_to_numpy(tensor):
    """Detach a tensor from the autograd graph, move it to CPU, and convert to a numpy array
    (for plotting/visualization)."""
    return tensor.detach().cpu().numpy()


# https://github.com/pytorch/pytorch/issues/32651
class SummaryWriter(tb.SummaryWriter):
    """Overrides `add_hparams` to log hparams/metrics into the *current* run's event file
    instead of creating a new subdirectory/run per call (the default `SummaryWriter`
    behaviour, tracked as a long-standing issue - see the linked thread above), so that
    hyperparameters logged mid-run show up alongside that run's other logged values."""
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None,
                    global_step: int = None):
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_dict (dict): Each key-value pair in the dictionary is the
              name of the hyperparameter and its corresponding value.
              The type of the value can be one of `bool`, `string`, `float`,
              `int`, or `None`.
            metric_dict (dict): Each key-value pair in the dictionary is the
              name of the metric and its corresponding value. Note that the key used
              here should be unique in the tensorboard record. Otherwise, the value
              you added by ``add_scalar`` will be displayed in hparam plugin. In most
              cases, this is unwanted.
            hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
              contains names of the hyperparameters and all discrete values they can hold
            run_name (str): not used
            global_step (Optional[int]): Global step of the run, to be included as part of the logdir.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

        Expected result:

        .. image:: _static/img/tensorboard/add_hparam.png
           :scale: 50 %

        """
        # noinspection PyUnresolvedReferences,PyProtectedMember
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = tb.summary.hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step=global_step)



class TBLogger(TensorBoardLogger):
    """PyTorch Lightning `TensorBoardLogger` that lazily creates its underlying writer as
    the patched `SummaryWriter` above instead of the default one, so hparams logged during
    training land in the current run rather than a new one."""
    @property
    def experiment(self):
        """Lazily construct (and cache) the underlying `SummaryWriter`."""
        if self._experiment is None:
            self._experiment = SummaryWriter(
                log_dir=self.log_dir,
                flush_secs=5,
            )
        return self._experiment

