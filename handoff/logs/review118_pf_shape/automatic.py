"""Let the Cartesian Gaussian learn velocity through position measurements."""

from contextlib import contextmanager
import rbpf_ctrv as rb
from cartesian import CartesianPF


class AutomaticCartesianPF(CartesianPF):
    def initialize_motion(self, mean, covariance, observation, n_frames, ref_pos):
        # The position update has already updated vx and vy through cross-covariance.
        mean, covariance = self.estimate()
        return mean, covariance, True


@contextmanager
def installed(enabled=True):
    previous = rb.ConditionalPF
    if enabled:
        rb.ConditionalPF = AutomaticCartesianPF
    try:
        yield
    finally:
        rb.ConditionalPF = previous
