"""Isolated R107 heuristic. No installed or repository tracker file is modified."""
from contextlib import contextmanager
import numpy as np
from pf_ctrv import PFTrackView


def recenter_velocity(pf, particles, logw, velocity):
    """Translate Cartesian velocity, preserving weighted residuals and other columns."""
    weights = np.exp(logw - np.max(logw))
    weights /= weights.sum()
    cart = particles[:, pf.V, None] * np.column_stack(
        (np.cos(particles[:, pf.THETA]), np.sin(particles[:, pf.THETA])))
    cart += np.asarray(velocity) - weights @ cart
    result = particles.copy()
    result[:, pf.V] = np.hypot(cart[:, 0], cart[:, 1])
    result[:, pf.THETA] = np.arctan2(cart[:, 1], cart[:, 0])
    return result


@contextmanager
def installed(enabled=True):
    """Only patch this Python process, and restore original methods even on failure."""
    original_init, original_update = PFTrackView.__init__, PFTrackView.update
    events = []
    if not enabled:
        yield events
        return

    def init(view, pf, track):
        original_init(view, pf, track)
        view.r107_last_frame = int(track.frame_id)
        view.r107_after_gap = False

    def update(view, mean, covariance, measurement, confidence=None):
        track, pf = view.track, view.pf
        frame = int(track._pending_frame_id)
        gap = frame - view.r107_last_frame
        previous = track._last_obs_xy
        result = original_update(view, mean, covariance, measurement, confidence)
        if gap == 1 and view.r107_after_gap and previous is not None and track._motion_initialized:
            velocity = (np.asarray(measurement[:2]) - np.asarray(previous)) / pf.dt
            if np.isfinite(velocity).all() and np.linalg.norm(velocity) >= pf._min_speed_for_heading:
                assert track._k1_prior is None
                track.particles = recenter_velocity(pf, track.particles, track.logw, velocity)
                result = pf.estimate_state(track.particles, track.logw)
                events.append({"frame": frame, "track_id": int(track.track_id),
                               "vx": float(velocity[0]), "vy": float(velocity[1])})
        view.r107_after_gap = gap > 1
        view.r107_last_frame = frame
        return result

    PFTrackView.__init__, PFTrackView.update = init, update
    try:
        yield events
    finally:
        PFTrackView.__init__, PFTrackView.update = original_init, original_update
