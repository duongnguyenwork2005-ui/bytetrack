"""Track classes for the RBPF, using the normal registry without monkeypatching."""

from rbpf import ConditionalPF
from pf_ctrv import PFTrackView
from tracker_ctrv import PFByteTracker, PFSTrack
from ul_backend import TrackState


class RBPFTrackView(PFTrackView):
    def predict(self, mean, covariance, lost=None):
        if lost is None:
            lost = self.track.state != TrackState.Tracked
        result = self.track.rbpf.predict(lost=lost)
        self.track.sync_rbpf()
        return result

    def update(self, mean, covariance, measurement, confidence=None):
        result = self.track.rbpf.update(measurement, confidence)
        self.track.sync_rbpf()
        return result

    def try_initiate_from_motion(self, mean, covariance, measurement, n_frames=1.0, ref_pos=None):
        result = self.track.rbpf.initialize_motion(
            mean, covariance, measurement, n_frames, ref_pos
        )
        self.track.sync_rbpf()
        return result


class RBPFSTrack(PFSTrack):
    def sync_rbpf(self):
        self.particles = self.rbpf.full_particles()
        self.logw = self.rbpf.logw.copy()
        self._k1_prior = None

    def activate(self, pf, frame_id):
        super().activate(pf, frame_id)
        self.rbpf = ConditionalPF(pf, self.mean, self.covariance, self.particles, self.rng)
        self.kalman_filter = RBPFTrackView(pf, self)
        self.sync_rbpf()

    @classmethod
    def multi_predict(cls, tracks):
        for track in tracks:
            track.mean, track.covariance = track.kalman_filter.predict(
                track.mean, track.covariance, lost=track.state != TrackState.Tracked
            )


class RBPFByteTracker(PFByteTracker):
    track_class = RBPFSTrack
