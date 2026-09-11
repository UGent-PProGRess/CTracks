"""Base classes for CT projectors: CTModule, the common interface all forward-model projectors
implement, and SequentialCTModule, which chains several CTModules into the ordered pipeline used
to simulate/reconstruct a sinogram (e.g. particles + matrix + noise, applied one after another).
"""
import collections
from typing import TYPE_CHECKING
import torch
from torch import nn

from ctrex.utils.datasets import CTDatasetInMemory, CTDatasetSparse, CTDataset
from ctrex.optimization import torchtools as tt
from ctrex.utils.trajectories import Trajectory
if TYPE_CHECKING:
    from ctrex.sample_description.sample_component import SampleComponent, sm, tm


class CTModule(tt.OptimModule):
    """
    Projector with a registered ct_trajectory and sample_component.
    This is different from a Projector that, given a sample x, computes y´ = A(x).
    These projectors take (y, _views, sampled_projections) and modify y like y´ = A(y) (e.g. y´ = y + A(x)).
    Registering the sample_component x to the projector allows to compute some defaults like a shape for the volume.
    A CT simulation can contain more than one SampleProjector that may or may not require some parameters x,
    which is why it is more convenient for the CT simulation that these parameters x are part of the SampleProjector.
    The views are also passed in the forward function because SampleProjectors share a ct_trajectory,
    and you don't want to compute views multiple times. Todo: This could also be a default None
    """
    def __init__(self, sample_component, **optim_kwargs):
        super().__init__(**optim_kwargs)
        self.sample_component = sample_component  # type: SampleComponent
        self.ct_trajectory = None  # type: type(None) | Trajectory
        self.projection_grid = None

    @property
    def track_model(self):
        """Shortcut to ``self.sample_component.track_model``."""
        track_model = self.sample_component.track_model  # type: tm.TrackModel
        return track_model

    @property
    def shape_model(self):
        """Shortcut to ``self.sample_component.shape_model``."""
        shape_model = self.sample_component.shape_model  # type: sm.ShapeModel
        return shape_model

    def centers_time(self, sampled_projections):
        """Evaluate the track model at the projection times corresponding to
        ``sampled_projections``, returning the component's center position(s) over time."""
        track_time = self.ct_trajectory.projection_time(sampled_projections)
        centers_time = self.track_model(track_time)
        return centers_time

    def forward(self, sinogram, views, sampled_projections):
        """Apply this projector to ``sinogram`` (e.g. ``sinogram' = sinogram + A(x)``) for the
        given ``views`` and ``sampled_projections``. Must be implemented by subclasses."""
        raise NotImplementedError

    def empty_sinogram(self, sino_params, sino_type = CTDatasetInMemory, roi = None, device = None, **kwargs):
        """Allocate an empty sinogram/dataset matching ``sino_params`` and ``ct_trajectory``,
        as a ``CTDatasetSparse``, ``CTDatasetInMemory``, or a plain zero tensor depending on
        ``sino_type``. Returns the resulting dataset (or tensor if ``sino_type`` is neither)."""
        num_projections = self.ct_trajectory.num_projections
        # num_projections = len(sino_params['angles'])
        projection_times = torch.arange(0, num_projections, dtype = torch.int, device = device)
        if sino_type == CTDatasetSparse:
            ct_dataset = CTDatasetSparse(sino_params, projection_times, device, **kwargs, roi = roi)
        else:
            height = 1 if sino_params['dimension'] == 2 else sino_params['height']
            width = sino_params['width']
            sinograms = torch.zeros((height, num_projections, width), dtype=torch.float32, device=device)
            if sino_type == CTDatasetInMemory:
                ct_dataset = CTDatasetInMemory(sino_params, projection_times, sinograms, roi = roi)
            else:
                return sinograms
        return ct_dataset  # type: CTDataset

    def set_ct_trajectory(self, ct_trajectory):
        """Register the ``Trajectory`` this projector should use for geometry (rays, views, ...)."""
        self.ct_trajectory = ct_trajectory

    @staticmethod
    def calc_vuwrite(v_particles, u_particles, v_grid, u_grid):
        """Round each particle's (sub-pixel) projected position to the nearest integer pixel and
        add the local patch grid offsets, giving the absolute (v, u) sinogram pixel coordinates to
        write each patch value into.

        Returns:
            u_int, v_int: rounded integer particle center coordinates.
            vu_write[1], vu_write[0]: absolute u/v write coordinates for every (particle, patch
                pixel) combination, i.e. rounded center + grid offset.
        """
        u_int, v_int = torch.round(u_particles), torch.round(v_particles)
        vu_write = (torch.stack((v_int, u_int), dim=0).unsqueeze(-1).unsqueeze(-1) +
                    torch.stack((v_grid, u_grid), dim=0).unsqueeze(1).unsqueeze(1)).int()
        return u_int, vu_write[1], v_int, vu_write[0]


class SequentialCTModule(nn.Sequential, CTModule):
    """An ordered chain of CTModule projectors applied one after another to build up a sinogram
    (like ``nn.Sequential`` but restricted to CTModule children, and forwarding
    ``(sinogram, views, sampled_projections)`` through each one instead of a single input).
    Accepts either an ``OrderedDict`` of name->module, or a plain sequence of modules (auto-named
    by class, de-duplicated with a numeric suffix)."""
    def __init__(self, *args, learning_rate = None):
        nn.Module.__init__(self)
        CTModule.__init__(self, None)
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key.replace(' ','_'), module)
        else:
            names = [module.__class__.__name__ for module in args]
            name_counter = {name: 0 for name in set(names)}
            for idx, module in enumerate(args):
                name = names[idx].replace(' ','_')
                self.add_module(name + str(name_counter[name]), module)
                name_counter[name] += 1
        if learning_rate is not None:
            self.learning_rates = learning_rate

    def __getitem__(self, idx) -> CTModule:
        """Index/slice into the child projectors, as with ``nn.Sequential``."""
        return super(SequentialCTModule, self).__getitem__(idx)

    def add_module(self, name, module):
        """Register a child ``module``, requiring it to be a ``CTModule``."""
        assert isinstance(module, CTModule)
        super().add_module(name, module)

    def projector_children(self) -> (str, CTModule):
        """Iterate only over registered child modules that are Projectors."""
        for name, module in super().named_children():
            if isinstance(module, CTModule):
                yield name, module

    def set_ct_trajectory(self, ct_trajectory):
        """ """
        for name, projector in self.projector_children():  # type: (str, CTModule)
            projector.set_ct_trajectory(ct_trajectory)

    # noinspection PyMethodOverriding
    def forward(self, sinogram, views, sampled_projections):
        """Apply each child projector to ``sinogram`` in registration order."""
        for name, module in self.projector_children():  # only children, should be ordered
            sinogram = module(sinogram, views, sampled_projections)
        return sinogram
