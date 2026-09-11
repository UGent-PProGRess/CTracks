"""Package re-exports for the projector classes that make up a CT forward model: the
CTModule/SequentialCTModule composition mechanism (base_modules), the ray-tracing simulations
(array_projectors, static_projectors), and the sinogram-domain projection effects
(projection_effects, e.g. CTNoise, HeelEffect).

Note: RayTraceCTSimulation is imported first from array_projectors and then immediately
re-imported (and shadowed) from static_projectors, so ``ctrex.projectors.RayTraceCTSimulation``
always resolves to the static_projectors version.
"""
from ctrex.projectors.base_modules import CTModule, SequentialCTModule
from ctrex.projectors.array_projectors import ArrayCTSimulation, ArrayIntersectionCTSimulation, RayTraceCTSimulation
from .static_projectors import RayTraceCTSimulation, SIRTCTSimulation
from ctrex.projectors.projection_effects import (CTNoise, HeelEffect, BeamFluctuations, CTEffects,
                                                 DetectorMask)

__all__ = ['CTModule',
           'SequentialCTModule',
           'ArrayCTSimulation',
           'ArrayIntersectionCTSimulation',
           'RayTraceCTSimulation',
           'SIRTCTSimulation',
           'CTNoise',
           'HeelEffect',
           'BeamFluctuations',
           'CTEffects',
           'DetectorMask']
