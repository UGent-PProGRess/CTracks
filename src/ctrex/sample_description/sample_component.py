"""Defines `SampleComponent`, pairing a track model (particle/volume positions over time) with a
shape model (their geometry/appearance) into one optimizable unit that can be projected by e.g.
`ParticleCTSimulation`/`SIRTCTSimulation`, and grown or shrunk by adding/removing particles."""

import torch
import copy

from ctrex.sample_description import track_models as tm, shape_models as sm
from ctrex.utils.trajectories import Trajectory
from ctrex.optimization.torchtools import OptimModule


class SampleComponent(OptimModule):
    """Pairs a `track_model` (governs particle/volume positions over time, see `track_models.py`)
    with a `shape_model` (governs their shape/appearance, see `shape_models.py`) as one
    `OptimModule`, so the pair can be optimized and passed as a unit into projector classes
    (`ParticleCTSimulation`, `SIRTCTSimulation`, ...). Also provides particle-count management:
    `add_particles`/`remove_particles` grow or shrink the component's particle set (and, if given
    an optimizer, keep its parameter groups and momentum state consistent), while
    `init_component`/`init_close` (re-)initialize its parameters from scratch or close to another
    component's."""

    def __init__(self, track_model, shape_model):
        super().__init__(None)
        self.track_model = track_model  # type: tm.TrackModel
        self.shape_model = shape_model  # type: sm.ShapeModel

    @property
    def num_particles(self):
        return self.track_model.num_particles

    def init_component(self, num_particles, init_mode_tracks = tm.init_random, init_mode_shapes = sm.init_random):
        """Initializes `num_particles` new particles from scratch, via `init_mode_tracks`/
        `init_mode_shapes` initialization strategies on the track/shape models respectively."""
        self.track_model.init_tracks(num_particles, init_mode_tracks)
        self.shape_model.init_shapes(num_particles, init_mode_shapes)

    def init_close(self, other_component, sigs_tracks, sigs_shapes):
        """Initializes this component's parameters close to `other_component`'s learned/
        ground-truth ones (`sigs_tracks`/`sigs_shapes` control the perturbation magnitude per
        model) - e.g. for reconstructions started near a known answer."""
        assert type(other_component) is SampleComponent
        self.track_model.init_close(next(other_component.track_model.parameters()), sigs_tracks) #TODO think about handling different parameter sets?
        self.shape_model.init_close(other_component.shape_model.shape_params, sigs_shapes)

    def extend(self, other_component, optimizer = None, state_keys = ("exp_avg", "exp_avg_sq")):
        """
        Add particles

        @param other_component:
        @param optimizer: Optional optimizer to update momentum of tracked parameter
        @param state_keys: Keys of the optimizer state dict
        @return:
        """
        assert isinstance(other_component, SampleComponent)
        assert type(other_component.track_model) is type(self.track_model)  # exact match
        assert type(other_component.shape_model) is type(self.shape_model)  # exact match

        # mirror loop in OptimModule.params_to_optimize() to get names correct
        for module_name, module in self.named_modules():
            if type(module) is Trajectory: continue
            for name, parameter in module.named_parameters(recurse = False):
                other_parameter = getattr(other_component, module_name).get_parameter(name)

                combined = torch.cat((parameter, other_parameter), 0)
                new_parameter = torch.nn.Parameter(combined)
                setattr(module, name, new_parameter)
                if optimizer is not None:
                    param_group = [pgroup for pgroup in optimizer.param_groups if name in pgroup.get('name', '')]
                    if param_group:
                        param_group[0]['params'] = [new_parameter]

                    if parameter in optimizer.state:
                        state = optimizer.state.pop(parameter)
                        for key in state_keys:
                            if key in state:
                                new_elements = torch.zeros(other_parameter.data.shape[0], *state[key].shape[1:],
                                                           dtype=state[key].dtype, device=state[key].device)
                                state[key] = torch.cat([state[key], new_elements], dim=0)
                        optimizer.state[new_parameter] = state

        return self

    def add_particles(self, num_particles, optimizer = None, init_mode_tracks = tm.init_random,  state_keys = ("exp_avg", "exp_avg_sq")):
        """Grows this component by `num_particles`, initialized via `init_mode_tracks` (e.g.
        inheriting velocity from nearby existing particles) and merged in via `extend`. If
        `optimizer` is given, mirrors the added parameters into its param groups/momentum state
        (see `extend`). No-ops if `num_particles == 0`."""
        if num_particles == 0: return self
        new_particles = copy.deepcopy(self)
        new_particles.track_model.pore_mask = self.track_model.pore_mask  # TODO copy.deepcopy sets pore mask to be the tensor only
        new_particles.init_component(num_particles, init_mode_tracks=init_mode_tracks)
        return self.extend(new_particles, optimizer, state_keys)

    def remove_particles(self, removal_mask, optimizer = None, state_keys = ("exp_avg", "exp_avg_sq")):
        """
        Remove particles

        @param removal_mask: Boolean mask of particles to be removed
        @param optimizer: Optional optimizer to update momentum of tracked parameter
        @param state_keys: Keys of the optimizer state dict

        @return:
        """
        if removal_mask.sum() == 0: return
        # mirror loop in OptimModule.params_to_optimize() to get names correct
        for module_name, module in self.named_modules():
            if type(module) is Trajectory: continue
            for name, parameter in module.named_parameters(recurse = False):
                remaining = parameter[~removal_mask]
                new_parameter = torch.nn.Parameter(remaining)
                setattr(module, name, new_parameter)
                if optimizer is not None:
                    param_group = [pgroup for pgroup in optimizer.param_groups if pgroup.get('name') == name]
                    if param_group:
                        param_group[0]['params'] = [new_parameter]
                    if parameter in optimizer.state:
                        state = optimizer.state.pop(parameter)
                        for key in state_keys:
                            if key in state:
                                state[key] = state[key][~removal_mask]
                        optimizer.state[new_parameter] = state
