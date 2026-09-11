
class ParticleUpdateModule:
    """Mixin providing shared interval-gating logic for heuristic (non-gradient) particle
    update callbacks. Subclasses are typically combined with ``lightning.Callback`` via
    multiple inheritance and call ``ParticleUpdateModule.__init__`` explicitly alongside
    ``Callback.__init__``.

    Any extra keyword arguments passed to the constructor are bound directly onto the
    instance as attributes (via ``setattr``) rather than declared explicitly, so subclasses
    can forward arbitrary config kwargs through ``**kwargs``.
    """
    def __init__(self, interval = None, **kwargs):
        self.interval = interval
        self.running = interval is not None and interval > 0
        self.run_iteration = False
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)

    def check_interval(self, iteration):
        """Return (and cache on self.run_iteration) whether an update should run this
        iteration: never on iteration 0, otherwise every `self.interval` iterations."""
        if iteration == 0 or not self.running:
            self.run_iteration = False  # don't do updates on first iteration
        else: self.run_iteration = iteration % self.interval == 0
        return self.run_iteration
    def disable_updates(self):
        """Permanently stop `check_interval` from returning True for this instance."""
        self.running = False

    def get_particle_component(self, ct_recon):
        """Fetch the particle SampleComponent from the reconstruction's ct_sample. Requires
        a ("particles", ParticleProjector) entry in the projectors OrderedDict."""
        assert hasattr(ct_recon.ct_sample.projectors,'particles'), 'Add a ("particles",ParticleProjector) to the ct_sample\'s projectors OrderedDict'
        return ct_recon.ct_sample.projectors.particles.sample_component